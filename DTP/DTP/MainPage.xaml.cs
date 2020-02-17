using DJI.WindowsSDK;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using System.Threading;
using DTP.Utils;
using Windows.Storage;
using Windows.Foundation.Metadata;
using Windows.ApplicationModel;
using Windows.UI.ViewManagement;

namespace DTP
{
    public sealed partial class MainPage : Page
    {
        private String CurrStrContentFrameName = null;

        public MainPage()
        {
            this.InitializeComponent();
            Connection.instance.isConnectionChangedEvent += MainPage_ConnectionChangedEvent;
            ControllerPage.isStopObjectTrackingEvent += MainPage_isStopObjectTrackingEvent;
            ApplicationView.PreferredLaunchViewSize =new Size(1600, 900);
            ApplicationView.PreferredLaunchWindowingMode = ApplicationViewWindowingMode.PreferredLaunchViewSize;

        }

      

        private async void MainPage_ConnectionChangedEvent(bool isConnected)
        {
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                if (isConnected)
                {
                    foreach (var item in NavView.MenuItems.OfType<NavigationViewItem>())
                    {
                      if(item.Tag.Equals("Controller_Page")) {
                            item.IsEnabled = true;
                        }
                    }
                }
                else
                {
                    foreach (var item in NavView.MenuItems.OfType<NavigationViewItem>())
                    {
                        if (item.Tag.Equals("Controller_Page"))
                        {
                            item.IsEnabled = false;
                        }
                    }
                }
            });
        }

        private async void MainPage_isStopObjectTrackingEvent()
        {
            

            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.High, async () =>
            {
                ContentFrame.Navigate(typeof(HomePage));

                foreach (var item in NavView.MenuItems.OfType<NavigationViewItem>())
                {
                    if (item.Tag.Equals("Home_Page"))
                    {
                        NavView.SelectedItem = item;
                        CurrStrContentFrameName = "HomePage";
                    }
                }
            });
        }

        private void NavView_ItemInvoked(NavigationView sender, NavigationViewItemInvokedEventArgs args)
        {

            String strItemTag = args.InvokedItem as String;
            
            if (strItemTag != null)
            {
                switch (strItemTag)
                {
                    case "Home":
                        if (CurrStrContentFrameName != "HomePage")
                        {
                            ContentFrame.Navigate(typeof(HomePage));
                            CurrStrContentFrameName = "HomePage";

                        }
                        break;


                    case "Controller":
                        if (CurrStrContentFrameName != "ControllerPage")
                        {
                            ContentFrame.Navigate(typeof(ControllerPage));
                            CurrStrContentFrameName = "ControllerPage";

                        }
                        break;
                }
            }
        }

        private void NavView_Loaded(object sender, RoutedEventArgs e)
        {
            ContentFrame.Navigate(typeof(HomePage));
            CurrStrContentFrameName = "HomePage";

        }

    }
}


