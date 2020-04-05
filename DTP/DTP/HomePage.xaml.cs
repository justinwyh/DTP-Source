using System;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media;
using DTP.Utils;
using Windows.UI;
using DJI.WindowsSDK;
using DJI.WindowsSDK.Components;
using DTP.Utils;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=234238

namespace DTP
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to  within a Frame.
    /// </summary>
    public sealed partial class HomePage : Page
    {
        public HomePage()
        {
            this.InitializeComponent();
            Connection.instance.isConnectionChangedEvent += HomePage_ConnectionChangedEvent;
            Connection.instance.isRegisterErrorOccurredEvent += HomePage_isRegisterErrorOccurredEvent;
            Controller.IsResultReceivedEvent += HomePage_isResultReceivedEvent;
            //set localsettings variables
            appDataPathTB.Text = @"C:\Users\Family\DTP_Data";
            APPConfig.instance.setConfigProperties("DataPath", appDataPathTB.Text.ToString().Replace(@"\",@"/")); 
            string[] filePaths = appDataPathTB.Text.ToString().Split(":\\");
            APPConfig.instance.setConfigProperties("Drive", filePaths[0] + ":\\");
            APPConfig.instance.setConfigProperties("ExtractedFramesFolder", filePaths[1] + @"\DJI\Output");

           
        }

        private void HomePage_isResultReceivedEvent(string s)
        {
            UpdateLogOutput(s);
        }

        private void HomePage_isRegisterErrorOccurredEvent(string s)
        {
            UpdateLogOutput(s);

        }

        private async void HomePage_ConnectionChangedEvent(Boolean isConnected)
        {
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                if (isConnected)
                {
                    //You can load/display your pages according to the aircraft connection state here.
                    ConnectionStatus.Text = "Connected";
                    ConnectionStatus.Foreground = new SolidColorBrush(Colors.Green);
                    ConnectionOutputTB.Text += (System.DateTime.Now + " The Aircraft is connected now." + Environment.NewLine);
                }
                else
                {

                    //You can hide your pages according to the aircraft connection state here, or show the connection tips to the users.
                    ConnectionStatus.Text = "Not Connected";
                    ConnectionStatus.Foreground = new SolidColorBrush(Colors.Red);
                    ConnectionOutputTB.Text += (System.DateTime.Now + " The Aircraft is disconnected now." + Environment.NewLine);
                }
            });

        }

        private async void ConnectionOutputTB_TextChanged(object sender, TextChangedEventArgs e)
        {
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                var grid = (Grid)VisualTreeHelper.GetChild(ConnectionOutputTB, 0);
                for (var i = 0; i <= VisualTreeHelper.GetChildrenCount(grid) - 1; i++)
                {
                    object obj = VisualTreeHelper.GetChild(grid, i);
                    if (!(obj is ScrollViewer)) continue;
                    ((ScrollViewer)obj).ChangeView(0.0f, ((ScrollViewer)obj).ExtentHeight, 1.0f);
                    break;
                }
            });
        }





        private async void UpdateLogOutput(String s)
        {
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                ConnectionOutputTB.Text += (System.DateTime.Now + " " + s + Environment.NewLine);

            });
        }

        private async void appDataPathTB_TextChanged(object sender, TextChangedEventArgs e)
        {
            APPConfig.instance.setConfigProperties("DataPath", appDataPathTB.Text.ToString().Replace(@"\", @"/"));
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                ConnectionOutputTB.Text += (System.DateTime.Now + " " + "DTP Data Path has been updated. Path: " + appDataPathTB.Text.ToString().Replace(@"\", @"/") + Environment.NewLine);

            });
        }
    }
}
