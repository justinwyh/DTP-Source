using System;
using Windows.UI.Xaml.Controls;

namespace DTP.Utils
{
    class UIOperations
    {

        private static ContentDialog ErrorDialog;
        private static String ErrorDialogContent = "";

        public async static void ShowContentDialog(string title, string s)
        {
            if (ErrorDialog != null)
            {
                ErrorDialog.Hide();
                ErrorDialog = null;
            }
            ErrorDialogContent += s + Environment.NewLine;
            ErrorDialog = new ContentDialog()
            {
                Title = title,
                Content = ErrorDialogContent,
                CloseButtonText = "OK"
            };
            
            await ErrorDialog.ShowAsync();
            ErrorDialog.Closed += ErrorDialogClosedEvent;

        }

        private static void ErrorDialogClosedEvent(ContentDialog sender, ContentDialogClosedEventArgs args)
        {
            ErrorDialogContent = "";
        }

    }
}
