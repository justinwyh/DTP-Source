using DJI.WindowsSDK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DTP.Utils
{
    sealed class Connection

    {
        public static readonly Connection instance = new Connection();
        public delegate void isErrorOccurred(String s);
        public event isErrorOccurred isRegisterErrorOccurredEvent;
        public delegate void isConnectionChanged(Boolean isConnected);
        public event isConnectionChanged isConnectionChangedEvent;
        public delegate void isProductTypeChanged(object sender, ProductTypeMsg? value);

        private Connection()
        {
            DJISDKManager.Instance.SDKRegistrationStateChanged += Connection_SDKRegistrationEvent;

            //Replace with your registered App Key. Make sure your App Key matched your application's package name on DJI developer center.
            DJISDKManager.Instance.RegisterApp("8981a69648b95f6b6b7ace4e");

        }

        private void Connection_SDKRegistrationEvent(SDKRegistrationState state, SDKError resultCode)
        {
            if (resultCode == SDKError.NO_ERROR)
            {
                //System.Diagnostics.Debug.WriteLine("Register app successfully.");
                //The product connection state will be updated when it changes here.
                //DJISDKManager.Instance.ComponentManager.GetProductHandler(0).ProductTypeChanged += async delegate (object sender, ProductTypeMsg? value)

                DJISDKManager.Instance.ComponentManager.GetProductHandler(0).ProductTypeChanged += Connection_isProductTypeChangedEvent;
              
            }
            else
            {
                isRegisterErrorOccurredEvent("Register SDK failed, the error is: " + resultCode.ToString());
            }
        }

        public void Connection_isProductTypeChangedEvent(object sender, ProductTypeMsg? value)
        {
            {

                if (value != null && value?.value != ProductType.UNRECOGNIZED)
                {
                    //System.Diagnostics.Debug.WriteLine("The Aircraft is connected now.");
                    isConnectionChangedEvent(true);
                }
                else
                {
                    //System.Diagnostics.Debug.WriteLine("The Aircraft is disconnected now.");
                    isConnectionChangedEvent(false);
                }
            };

        }
        async public void isAirCraftConnected()
        {
            try
            {
                var productType = (await DJISDKManager.Instance.ComponentManager.GetProductHandler(0).GetProductTypeAsync()).value;
                if (productType != null && productType?.value != ProductType.UNRECOGNIZED)
                {
                    //System.Diagnostics.Debug.WriteLine("The Aircraft is connected now.");
                    isConnectionChangedEvent(true);
                }
                else
                {
                    isConnectionChangedEvent(false);
                }
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("isAirCraftConnected Error", ex.ToString());
            }

        }
    }
}
