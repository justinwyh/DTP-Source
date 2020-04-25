using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using Windows.ApplicationModel;
using Windows.ApplicationModel.AppService;
using Windows.Foundation.Collections;
using TrackingWPF.Utils;
using System.Text.RegularExpressions;

namespace TrackingWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string ProgramPath = @"C:\Users\Family\Anaconda3\envs\pysot\python.exe";
        private PythonProgram PyObjectTracking;
        private AppServiceConnection Connection = null;
        private ConcurrentQueue<string> PyOutputQueue = new ConcurrentQueue<string>();
        private ConcurrentQueue<string> PyConsoleOutputQueue = new ConcurrentQueue<string>();
        private ConcurrentQueue<ValueSet> MessageQueue = new ConcurrentQueue<ValueSet>();
        private Boolean IsHandlePyOutputTaskEnabled = true;
        private Boolean IsSendMessageTaskEnabled = true;
        private Boolean IsConsoleOutputTaskEnabled = true;
        private int buffersize;
        private int Buffersize { get => buffersize; set => buffersize = value; }

        //private string loggingpath = "C:\\Users\\Family\\Desktop\\log" + DateTime.Now.Second + ".txt"; //For Debug

        public MainWindow()
        {
            InitializeComponent();
            StartTBConsoleWriteLineTask();
            StartHandlePyOutputTask();
            StartSendBoundingBoxesTask();
            InitAppServiceConnection();
            PythonPathTBox.Text = ProgramPath;
            buffersize = 1;
        }

        private void HandlePyOutputTask()
        {
            while (IsHandlePyOutputTaskEnabled)
            {
                if (PyOutputQueue.IsEmpty)
                {
                    Thread.Sleep(5);
                }
                else
                {
                    string OutputString = null;
                    if (PyOutputQueue.TryDequeue(out OutputString))
                    {
                        try
                        {

                            if (OutputString.Contains("PYSOT_FPS"))
                            {
                                double Fps = Double.Parse(OutputString.Replace("PYSOT_FPS", ""));
                                UpdateOTFPSLabel(Fps);
                            }
                            else if (OutputString.Contains("PYSOT_BBOX"))
                            {
                                string[] Bbox = OutputString.Replace("PYSOT_BBOX", "").Replace("[", "").Replace("]", "").Replace(" ", "").Split(',');
                                int Bbox_XCorr = Int32.Parse(Bbox[0]);
                                int Bbox_YCorr = Int32.Parse(Bbox[1]);
                                int Bbox_XLength = Int32.Parse(Bbox[2]);
                                int Bbox_YLength = Int32.Parse(Bbox[3]);

                                ValueSet request = new ValueSet();
                                request.Add("Bbox_XCorr", Bbox_XCorr);
                                request.Add("Bbox_YCorr", Bbox_YCorr);
                                request.Add("Bbox_XLength", Bbox_XLength);
                                request.Add("Bbox_YLength", Bbox_YLength);
                                MessageQueue.Enqueue(request);
                            }
                            //else if (OutputString.Contains("JPS_FrameQueue_Size"))
                            //{
                            //    int Frame_Queue_Size = int.Parse(OutputString.Replace("JPS_FrameQueue_Size", ""));
                            //}
                            else if (OutputString.Contains("StartReceiveJPGStream"))
                            {
                                bool isStartProduceJPG = bool.Parse(OutputString.Replace("StartReceiveJPGStream", ""));

                                ValueSet request = new ValueSet();
                                request.Add("isStartProduceJPG", isStartProduceJPG);

                                MessageQueue.Enqueue(request);
                            }
                            else if (OutputString.Contains("Frame_Size"))
                            {
                                string[] height_width = OutputString.Replace("Frame_Size ", "").Split(' ');
                                int frameHeight = int.Parse(height_width[0]);
                                int frameWidth = int.Parse(height_width[1]);
                                ValueSet request = new ValueSet();
                                request.Add("frame_height", frameHeight);
                                request.Add("frame_width", frameWidth);
                                MessageQueue.Enqueue(request);
                            }
                            //PyConsoleOutputQueue.Enqueue(OutputString);  //For Debug
                            else if (!OutputString.Contains("Debug"))
                            {
                                PyConsoleOutputQueue.Enqueue(OutputString);
                            }
                        }
                        catch (Exception ex)
                        {
                            PyConsoleOutputQueue.Enqueue(ex.ToString() +'\n' + "OutputString:" + OutputString);
                        }
                    }
                }
            }
        }

        private async void SendMessageTask()
        {
            while (IsSendMessageTaskEnabled)
            {
                if (MessageQueue.IsEmpty)
                {
                    Thread.Sleep(5);
                }
                else
                {
                    ValueSet TmpRequest;
                    if (MessageQueue.TryDequeue(out TmpRequest))
                    {
                        try
                        {
                            AppServiceResponse response = await Connection.SendMessageAsync(TmpRequest);
                        }
                        catch (Exception ex)
                        {
                            PyConsoleOutputQueue.Enqueue(ex.ToString());
                        }
                    }
                }
            }
        }

        private void StartSendBoundingBoxesTask()
        {
            var SendBoundingBoxesTaskThread = new Thread(new ThreadStart(SendMessageTask));
            SendBoundingBoxesTaskThread.Start();
        }

        private void StartHandlePyOutputTask()
        {
            var HandlePyOutputThread = new Thread(new ThreadStart(HandlePyOutputTask));
            HandlePyOutputThread.Start();
        }

        private void StartTBConsoleWriteLineTask()
        {
            var TBConsoleWriteLineTaskThread = new Thread(new ThreadStart(TBConsoleWriteLineTask));
            TBConsoleWriteLineTaskThread.Start();
        }

        private void PyOTConsoleOutputEvent(string OutputString)
        {
            if (IsHandlePyOutputTaskEnabled)
            {
                if (!Regex.Replace(OutputString, @"\s+", "").Equals(""))
                {
                    PyOutputQueue.Enqueue(OutputString);
                }      
            }
        }

        //UI Controls
        void TBConsoleWriteLineTask()
        {

            {
                while (IsConsoleOutputTaskEnabled)
                {
                    if (!PyConsoleOutputQueue.IsEmpty)
                    {
                        string str = null;
                        PyConsoleOutputQueue.TryDequeue(out str);
                        if (!string.IsNullOrEmpty(str))
                        {
                            Application.Current.Dispatcher.Invoke(DispatcherPriority.DataBind, new Action(() =>
                                {
                                    pyoutputTB.Text += (str + Environment.NewLine);
                                    pyoutputTB.Focus();
                                    pyoutputTB.CaretIndex = pyoutputTB.Text.Length;
                                    pyoutputTB.ScrollToEnd();
                                }
                            ));

                            //File.AppendAllText(loggingpath, str + Environment.NewLine); //For Debug

                        }
                    }
                    else
                    {
                        Thread.Sleep(5);
                    }
                }

            }
        }



        async void UpdateOTFPSLabel(double fps)
        {
            await Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Normal,
                new Action(() =>
                {
                    OTFPSlbl.Content = fps;
                }));
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            if (File.Exists(PythonPathTBox.Text) && PythonPathTBox.Text.Contains("python.exe"))
            {
                PyOTConsoleOutputEvent(PythonPathTBox.Text);
                PyObjectTracking = new PythonProgram("Object Tracking", PythonPathTBox.Text);
                PyObjectTracking.tbORHandler += PyOTConsoleOutputEvent;
                
                try
                {
                    //APPConfig.instance.setConfigProperties("DataPath", @"C:/Users/Family/DTP_Data"); //FOR DEBUG

                    String AppDataPath = APPConfig.instance.getConfigProperties("DataPath");
                    //String AppDataPath = @"C:/Users/Family/DTP_Data";  //For Debug

                    String PyParams = AppDataPath + "/Source/Main.py --pysotconfig " + AppDataPath + "/Source/PySOT/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml  " +
                                                           "--pysotcheckpoint " + AppDataPath + "/Source/PySOT/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth " +
                                                             "--data ";
                    switch (NightModeTButton.IsChecked)
                    {
                        case true:
                            PyParams += AppDataPath + "/DJI/Output --buffer_size " + Buffersize +  " --dark_mode " + "--dbllnetcheckpoint " + AppDataPath + "/Source/DBLLNet/checkpoint/dbllnet_cm1ms-ssim.pth";
                            break;
                        case false:
                            PyParams += AppDataPath + "/DJI/Output --buffer_size " + Buffersize + " --no-dark_mode";
                            break;
                    }


                    PyOTConsoleOutputEvent(PyParams);

                    PyObjectTracking.executePythonProgram(PyParams);
                }
                catch(Exception ex)
                {
                    PyOTConsoleOutputEvent(ex.ToString());
                }

                await Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Normal, new Action(() =>
                {
                    StartButton.IsEnabled = false;
                }));
            }


        }


        private async void NightModeTButton_Click(object sender, RoutedEventArgs e)
        {
            await Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Normal,
                new Action(() =>
                {
                    switch (NightModeTButton.IsChecked)
                    {
                        case true:
                            NightModeTButton.Content = "ON";
                            break;
                        case false:
                            NightModeTButton.Content = "OFF";
                            break;
                    }
                }));
        }

        private async void InitAppServiceConnection()
        {
            try
            {
                Connection = new AppServiceConnection();
                Connection.AppServiceName = "DTPInteropAppService";
                Connection.PackageFamilyName = Package.Current.Id.FamilyName;
                Connection.ServiceClosed += ASC_ServiceClosed;

                AppServiceConnectionStatus status = await Connection.OpenAsync();
                if (status != AppServiceConnectionStatus.Success)
                {
                    MessageBox.Show(status.ToString());
                    this.IsEnabled = false;
                }
                else
                {
                    PyConsoleOutputQueue.Enqueue("Initiate App Service Success");
                }
            }
            catch (Exception ex)
            {
                PyConsoleOutputQueue.Enqueue(ex.ToString());
            }
        }


        private void ASC_ServiceClosed(AppServiceConnection sender, AppServiceClosedEventArgs args)
        {
            IsHandlePyOutputTaskEnabled = false;

        }

        private void FrameBufferSizeTB_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (FrameBufferSizeTB.Text != "")
            {
                Buffersize = int.Parse(FrameBufferSizeTB.Text);

            }
        }
    }
}