using DJI.WindowsSDK;
using DJI.WindowsSDK.Components;
using DJIVideoParser;
using DTP.Utils;
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Windows.ApplicationModel;
using Windows.ApplicationModel.AppService;
using Windows.Foundation.Metadata;
using Windows.Graphics.Imaging;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

namespace DTP
{
    public sealed partial class ControllerPage : Page
    {
        private Parser videoParser;
        private CameraHandler cameraHandler = DJISDKManager.Instance.ComponentManager.GetCameraHandler(0, 0);
        private string outputDrivePath = APPConfig.instance.getConfigProperties("Drive");
        private string outputFolderPath = APPConfig.instance.getConfigProperties("ExtractedFramesFolder");
        private StorageFolder ExtractedFramesFolder;

        private int imageCount;
        private int framesRendered = 0;
        private int fps = 0;
        //private int received_Bbox_Count = 0;
        private System.DateTime lastTime;

        private ConcurrentQueue<Tuple<byte[], int, int, int>> DecodedImageBytesQueue = new ConcurrentQueue<Tuple<byte[], int, int, int>>();
        private Boolean IsProdcuceImageTaskEnabled = false;
        private Boolean IsStopThread = false;

        private ObjectTracking objectTracking = new ObjectTracking();
        public delegate void isStopObjectTracking();
        public static event isStopObjectTracking isStopObjectTrackingEvent;

        public ControllerPage()
        {
            this.InitializeComponent();
            DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).VelocityChanged += ControllerPage_VelocityChanged;
            DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).AttitudeChanged += ControllerPage_AttitudeChanged;
            DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).AltitudeChanged += ControllerPage_AltitudeChanged;
            Connection.instance.isConnectionChangedEvent += ControllerPage_ConnectionChangedEvent;
            Controller.IsResultReceivedEvent += ControllerPage_isResultReceivedEvent;
            Window.Current.CoreWindow.KeyDown += MainGrid_KeyDown;
            Window.Current.CoreWindow.KeyUp += MainGrid_KeyUp;
        }

       

        protected override async void OnNavigatedTo(NavigationEventArgs e)
        {
            try
            {
                base.OnNavigatedFrom(e);
                await Init();
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }

        }

        protected async override void OnNavigatedFrom(NavigationEventArgs e)
        {
            try
            {
                Connection.instance.isConnectionChangedEvent -= ControllerPage_ConnectionChangedEvent;
                base.OnNavigatedTo(e);
                await UninitializeVideoFeedModule();
                await UninitAsync();
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("ControllerPage OnNavigatedFrom Error", ex.ToString());
            }

        }

        private async void ControllerPage_ConnectionChangedEvent(bool isConnected)
        {
            if (!isConnected)
            {
                await UninitAsync();
                isStopObjectTrackingEvent();

            }
        }





        private async Task Init()
        {
            try
            {
                //Step 1: Housekeep Output Folder and Create a new Output Folder
                await FileOperations.HousekeepExtractedFramesFolder();
                await FileOperations.CreateApplicationDataFolder();
                await GetFolderObjectAndCreateTmpFolder();

                //Step 2: Start Threaded Tasks
                StartProduceImageFromBytesTask();
                objectTracking.StartIssueControlSignalTask();

                //Step 3: Initalise Video Feed
                await InitializeVideoFeedModule();

                //Step 4: Initalise Camera Work Mode
                var current = await cameraHandler.GetCameraWorkModeAsync();
                var currentMode = current.value?.value;
                if (currentMode != CameraWorkMode.PLAYBACK && currentMode != CameraWorkMode.TRANSCODE)
                {
                    var returnCode = await cameraHandler.SetCameraWorkModeAsync(new CameraWorkModeMsg { value = CameraWorkMode.TRANSCODE });
                }
                else
                {
                    var returnCode = await cameraHandler.SetCameraWorkModeAsync(new CameraWorkModeMsg { value = CameraWorkMode.SHOOT_PHOTO });
                }
                var returnCode2 = await cameraHandler.SetCameraWorkModeAsync(new CameraWorkModeMsg { value = CameraWorkMode.SHOOT_PHOTO });

                //Step 5: Initialise aircraft settings
                await Controller.SetAircraftLimitation();
                await Controller.SetAircraftFailSafeAction();
                await Controller.EnableLandingProtection();
                await Controller.EnableObstacleAvoidance();
                await Controller.EnableUpwardsAvoidance();
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }

        }

        private async Task UninitAsync()
        {
            try
            {
                //Step 1: Stop all Tasks
                EndTask();
                //Step 2: Send stop and land command in case the drone has not landed
                Boolean isFlying = await Controller.IsFlyingAsync();
                if (isFlying)
                {
                    Controller.UpdateJoyStickValue("");
                    Controller.StartAutoLanding();
                }
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }

        }

        private void EndTask()
        {
            IsProdcuceImageTaskEnabled = false;
            IsStopThread = true;
            objectTracking.IsIssueControlSignalTaskEnabled = false;
            objectTracking.IsStopThread = true;
        }







        // UI Control

        private async void UpdateConsoleOuptut(String s)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                ConsoleOutputTB.Text += (System.DateTime.Now + " " + s + Environment.NewLine);
            });
        }

        private async void UpdateUIFPSValue(int fps)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                FPSlbl.Text = fps.ToString();
            });
        }


        private async void BBoxOutputTB_TextChanged(object sender, TextChangedEventArgs e)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                var grid = (Grid)VisualTreeHelper.GetChild(ConsoleOutputTB, 0);
                for (var i = 0; i <= VisualTreeHelper.GetChildrenCount(grid) - 1; i++)
                {
                    object obj = VisualTreeHelper.GetChild(grid, i);
                    if (!(obj is ScrollViewer)) continue;
                    ((ScrollViewer)obj).ChangeView(0.0f, ((ScrollViewer)obj).ExtentHeight, 1.0f);
                    break;
                }
            });
        }

        private void maeTB_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (maeTB.Text != "") {
                objectTracking.AcceptableError = double.Parse(maeTB.Text) / 100;

            }
        }

        private void ATOButton_Click(object sender, RoutedEventArgs e)
        {
            Controller.StartAutoTakeoff();
        }

        private void ALButton_Click(object sender, RoutedEventArgs e)
        {
            Controller.StartAutoLanding();
        }

        private void ControllerPage_isResultReceivedEvent(string result)
        {
            UpdateConsoleOuptut(result);
        }
        private void MainGrid_KeyDown(object sender, KeyEventArgs e)
        {
            Controller.isManualControlOverride = true;
            Controller.ControlJoyStickByKey(e.VirtualKey);
        }

        private void MainGrid_KeyUp(object sender, KeyEventArgs e)
        {
            Controller.isManualControlOverride = false;
            Controller.UpdateJoyStickValue("");
        }

        private async void VOTStartButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                StartTrackingWPF();
                await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
                {
                    VOTStartButton.IsEnabled = false;
                });
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }
        }

        private void VOTStopButton_Click(object sender, RoutedEventArgs e)
        {
            isStopObjectTrackingEvent();
        }







        //Display Velocity

        Velocity3D aircraftVelocity3D;
        public Velocity3D AircraftVelocity
        {
            get
            {
                return aircraftVelocity3D;
            }
            set
            {
                aircraftVelocity3D = value;
                xvelocitylbl.Text = AircraftVelocityXString;
                yvelocitylbl.Text = AircraftVelocityYString;
                zvelocitylbl.Text = AircraftVelocityZString;

            }
        }

        private async void ControllerPage_VelocityChanged(object sender, Velocity3D? value)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                if (value != null)
                {
                    AircraftVelocity = value.Value;

                }
            });
        }

        private String AircraftVelocityXString
        {
            get { return aircraftVelocity3D.x.ToString() + " m/s"; }
        }
        private String AircraftVelocityYString
        {
            get { return aircraftVelocity3D.y.ToString() + " m/s"; }
        }
        private String AircraftVelocityZString
        {
            get { return aircraftVelocity3D.z.ToString() + " m/s"; }
        }








        //Display Attitude

        Attitude attitude;

        public Attitude AircraftAttitude
        {
            get
            {
                return AircraftAttitude;
            }
            set
            {
                attitude = value;
                pitchlbl.Text = AircraftAttitudePitchString;
                rolllbl.Text = AircraftAttitudeRollString;
                yawlbl.Text = AircraftAttitudeYawString;
            }
        }

        private String AircraftAttitudePitchString
        {
            get { return attitude.pitch.ToString() + "\u00B0"; }
        }
        private String AircraftAttitudeRollString
        {
            get { return attitude.roll.ToString() + "\u00B0"; }
        }
        private String AircraftAttitudeYawString
        {
            get { return attitude.yaw.ToString() + "\u00B0"; }
        }

        private async void ControllerPage_AttitudeChanged(object sender, Attitude? value)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                if (value != null)
                {
                    AircraftAttitude = value.Value;

                }
            });
        }



        private async void ControllerPage_AltitudeChanged(object sender, DoubleMsg? value)
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
            {
                if (value != null)
                {
                    altitudelbl.Text = value.Value.value.ToString();

                }
            });
        }

      





        //  Windows Extensions and App Service Connections

        private async void StartTrackingWPF()
        {
            try
            {
                if (ApiInformation.IsApiContractPresent("Windows.ApplicationModel.FullTrustAppContract", 1, 0))
                {
                    App.AppServiceConnected += ContollerPage_AppServiceConnected;
                    App.AppServiceDisconnected += ControllerPage_AppServiceDisconnected;
                    await FullTrustProcessLauncher.LaunchFullTrustProcessForCurrentAppAsync();
                }
            }
            catch (Exception ex)
            {

                UpdateConsoleOuptut(ex.ToString());

            }

        }

        private void ContollerPage_AppServiceConnected(object sender, AppServiceTriggerDetails e)
        {
            UpdateConsoleOuptut("AppService - Connected to TrackingWPF process");
            App.AppServiceConnection.RequestReceived += AppServiceConnection_RequestReceived;
        }

        private void ControllerPage_AppServiceDisconnected(object sender, EventArgs e)
        {
            UpdateConsoleOuptut("AppService - Lost connection to TrackingWPF process");
        }

        private async void AppServiceConnection_RequestReceived(AppServiceConnection sender, AppServiceRequestReceivedEventArgs args)
        {
            try
            {

                if (args.Request.Message.ContainsKey("frame_height"))
                {
                    objectTracking.Frame_height = (int)args.Request.Message["frame_height"];
                    objectTracking.Frame_width = (int)args.Request.Message["frame_width"];
                    objectTracking.IsIssueControlSignalTaskEnabled = true;

                }

                else if (args.Request.Message.ContainsKey("isStartProduceJPG"))
                {
                    IsProdcuceImageTaskEnabled = Boolean.Parse(args.Request.Message["isStartProduceJPG"].ToString());

                    if (!IsProdcuceImageTaskEnabled)
                    {
                        IsStopThread = true;
                        objectTracking.IsIssueControlSignalTaskEnabled = false;
                        objectTracking.IsStopThread = true;
                        Boolean isFlying = await Controller.IsFlyingAsync();
                        if (isFlying)
                        {
                            Controller.UpdateJoyStickValue("");
                            Controller.StartAutoLanding();
                        }
                    }
                }

                else if(args.Request.Message.ContainsKey("Bbox_XCorr"))
                {
                    int Xcorr = (int)args.Request.Message["Bbox_XCorr"];
                    int Ycorr = (int)args.Request.Message["Bbox_YCorr"];
                    int Xlength = (int)args.Request.Message["Bbox_XLength"];
                    int Ylength = (int)args.Request.Message["Bbox_YLength"];
                    int[] BoundingBoxValues = { Xcorr, Ycorr, Xlength, Ylength };
                    //UpdateConsoleOuptut("AppService - Received 1 BoundingBoxValues");  //For Debug
                    objectTracking.BoundingBoxQueue.Enqueue(BoundingBoxValues);
                    //UpdateUIBbCountValue(++received_Bbox_Count);
                }
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }
        }



        ///DJI Video Feed Handling Methods

        private async Task InitializeVideoFeedModule()
        {
            //Must in UI thread
            await Dispatcher.RunAsync(CoreDispatcherPriority.High, async () =>
            {
                try
                {
                    //Raw data and decoded data listener
                    if (videoParser == null)
                    {
                        videoParser = new Parser();
                        videoParser.Initialize(delegate (byte[] data)
                        {
                            //Note: This function must be called because we need DJI Windows SDK to help us to parse frame data.
                            return DJISDKManager.Instance.VideoFeeder.ParseAssitantDecodingInfo(0, data);
                        });
                        //Set the swapChainPanel to display and set the decoded data callback.
                        videoParser.SetSurfaceAndVideoCallback(0, 0, swapChainPanel, ReceiveDecodedData);
                        DJISDKManager.Instance.VideoFeeder.GetPrimaryVideoFeed(0).VideoDataUpdated += OnVideoPush;
                    }
                    //get the camera type and observe the CameraTypeChanged event.
                    DJISDKManager.Instance.ComponentManager.GetCameraHandler(0, 0).CameraTypeChanged += OnCameraTypeChanged;
                    var type = await DJISDKManager.Instance.ComponentManager.GetCameraHandler(0, 0).GetCameraTypeAsync();
                    OnCameraTypeChanged(this, type.value);
                }
                catch (Exception ex)
                {
                    ConsoleOutputTB.Text += (System.DateTime.Now.ToString() + " " + ex.ToString());
                }
                finally
                {
                     imageCount = 0;
                }
            });
        }


        private async Task UninitializeVideoFeedModule()
        {
            try
            {
                if (DJISDKManager.Instance.SDKRegistrationResultCode == SDKError.NO_ERROR)
                {
                    //videoParser.SetSurfaceAndVideoCallback(0, 0, null, null);
                    DJISDKManager.Instance.VideoFeeder.GetPrimaryVideoFeed(0).VideoDataUpdated -= OnVideoPush;
                    await FileOperations.HousekeepExtractedFramesFolder();

                }
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }
        }


        void OnVideoPush(VideoFeed sender, byte[] bytes)
        {
            //raw data
            videoParser.PushVideoData(0, 0, bytes, bytes.Length);
        }

        void ReceiveDecodedData(byte[] data, int width, int height)
        {
            //This function would return a bytes array with image data in RGBA format.
            if (IsProdcuceImageTaskEnabled)
            {
                imageCount++;
                DecodedImageBytesQueue.Enqueue(Tuple.Create(data, width, height, imageCount));
            }
        }

        private void ProduceImageFromBytesTask()
        {
            while (!IsStopThread)
            {
                while (IsProdcuceImageTaskEnabled)
                {
                    if (DecodedImageBytesQueue.IsEmpty)
                    {
                        Thread.Sleep(20);
                    }
                    else
                    {
                        Tuple<byte[], int, int, int> TempImgTuple = null;
                        if (DecodedImageBytesQueue.TryDequeue(out TempImgTuple))
                        {
                            ConvertyBytesToJPG(TempImgTuple.Item1, TempImgTuple.Item2, TempImgTuple.Item3, TempImgTuple.Item4);
                        }
                    }
                }
            }

        }

        private void StartProduceImageFromBytesTask()
        {
            var ImageFactoryThread = new Thread(new ThreadStart(ProduceImageFromBytesTask));
            ImageFactoryThread.Start();
        }

        async void ConvertyBytesToJPG(byte[] data, int width, int height, int currImageCount)
        {
            try
            {
                if (data != null)
                {
                    // Debug: For printing the byte data
                    //var tmpSB = new StringBuilder("Size:");
                    //tmpSB.Append(data.Length + " { ");
                    //foreach (var b in data)
                    //{
                    //    tmpSB.Append(b + ", ");
                    //}
                    //tmpSB.Append("}");
                    //StorageFile tmpLogTxt = await ExtractedFramesFolder.CreateFileAsync("Log" + currImageCount.ToString() + ".txt", CreationCollisionOption.OpenIfExists);
                    //await FileIO.AppendTextAsync(tmpLogTxt, tmpSB.ToString());
                    //await FileIO.AppendTextAsync(tmpLogTxt, Environment.NewLine);
                    //System.Diagnostics.Debug.WriteLine(currImageCount);

                    //Generate JPG files for OpenCV in Python to consume
                    StorageFile tempFile = await ExtractedFramesFolder.CreateFileAsync((currImageCount).ToString() + ".jpg", CreationCollisionOption.ReplaceExisting);
                    var raStream = await tempFile.OpenStreamForWriteAsync();
                    var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, raStream.AsRandomAccessStream());
                    encoder.SetPixelData(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Straight, (uint)width, (uint)height, 96, 96, data);
                    await encoder.FlushAsync();


                    //System.Diagnostics.Debug.WriteLine("Received image count:" + currImageCount); //FOR DEBUG
                    CaculateFrameGenerationTime();
                }
            }
            catch (Exception ex)
            {

                UpdateConsoleOuptut(ex.ToString());

            }
        }

        //Util Funtions

        void CaculateFrameGenerationTime()
        {
            framesRendered++;

            if ((System.DateTime.Now - lastTime).TotalSeconds >= 1)
            {

                fps = framesRendered;
                framesRendered = 0;
                lastTime = System.DateTime.Now;
                //UpdateConsoleOuptut("Streaming JPG Output FPS:" + fps); //For Debug
                UpdateUIFPSValue(fps);
            }
        }

        public async Task GetFolderObjectAndCreateTmpFolder()
        {
            try
            {
                ExtractedFramesFolder = await StorageFolder.GetFolderFromPathAsync(outputDrivePath + outputFolderPath);
            }
            catch (Exception ex)
            {
                UpdateConsoleOuptut(ex.ToString());
            }
        }

        //We need to set the camera type of the aircraft to the DJIVideoParser. After setting camera type, DJIVideoParser would correct the distortion of the video automatically.
        private void OnCameraTypeChanged(object sender, CameraTypeMsg? value)
        {
            if (value != null)
            {
                switch (value.Value.value)
                {
                    case CameraType.MAVIC_2_ZOOM:
                        this.videoParser.SetCameraSensor(AircraftCameraType.Mavic2Zoom);
                        break;
                    case CameraType.MAVIC_2_PRO:
                        this.videoParser.SetCameraSensor(AircraftCameraType.Mavic2Pro);
                        break;
                    default:
                        this.videoParser.SetCameraSensor(AircraftCameraType.Others);
                        break;
                }

            }
        }

      
    }


}


