using DJI.WindowsSDK;
using System;
using System.Threading.Tasks;
using Windows.System;

namespace DTP.Utils
{
    class Controller
    {
        public delegate void Controller_Page_ResultReceived(String s);

        public static event Controller_Page_ResultReceived IsResultReceivedEvent;

        public static float curr_roll, curr_pitch, curr_throttle, curr_yaw;

        public async static void StartAutoTakeoff()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).StartTakeoffAsync();
                IsResultReceivedEvent(String.Format("Start send takeoff command: {0}", res.ToString()));
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("StartAutoTakeoff Error", ex.ToString());
            }
        }

        public async static void StartAutoLanding()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).StartAutoLandingAsync();
                IsResultReceivedEvent(String.Format("Start send landing command: {0}", res.ToString()));
                //var res1 = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).GetIsLandingConfirmationNeededAsync();
                //isResultReceivedEvent(String.Format("Is Confirm landing Required: {0}", res1.value.Value.value.ToString()));

                //if (res1.value.Value.value)
                //{
                //    var res2 = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).ConfirmLandingAsync();
                //    isResultReceivedEvent("Confirm Landing Result: " + res2.ToString());
                //}
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("StartAutoLanding Error", ex.ToString());
            }

        }

        public static void UpdateJoyStickValue(String command)
        {
            float JoystickValue = 0.1F;
             switch (command) 
            {
                case "roll_left": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, 0, 0, -JoystickValue); break;
                case "roll_right": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, 0, 0, +JoystickValue); break;
                case "pitch_forward": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, 0, +JoystickValue, 0); break;
                case "pitch_backward": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, 0, -JoystickValue, 0); break;
                case "throttle_up": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(+JoystickValue, 0, 0, 0); break;
                case "throttle_down": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(-JoystickValue, 0, 0, 0); break;
                case "yaw_left": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, -JoystickValue, 0, 0); break;
                case "yaw_right": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, +JoystickValue, 0, 0); break;
                case "": DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(0, 0, 0, 0); break;
                default: break;

            }

        }


        public static void UpdateJoyStickValue()
        {
            DJISDKManager.Instance.VirtualRemoteController.UpdateJoystickValue(curr_throttle, curr_yaw, curr_pitch, curr_roll); 
        }

        public static void ControlJoyStickByKey(VirtualKey key)
        {
            switch (key)
            {
                //roll
                case VirtualKey.A: Controller.UpdateJoyStickValue("roll_left"); break;
                case VirtualKey.D: Controller.UpdateJoyStickValue("roll_right"); break;
                //pitch
                case VirtualKey.W: Controller.UpdateJoyStickValue("pitch_forward"); break;
                case VirtualKey.S: Controller.UpdateJoyStickValue("pitch_backward"); break;
                //Throttle
                case VirtualKey.I: Controller.UpdateJoyStickValue("throttle_up"); break;
                case VirtualKey.K: Controller.UpdateJoyStickValue("throttle_down"); break;
                //yaw
                case VirtualKey.J: Controller.UpdateJoyStickValue("yaw_left"); break;
                case VirtualKey.L: Controller.UpdateJoyStickValue("yaw_right"); break;
                //stop
                case VirtualKey.Space: Controller.UpdateJoyStickValue(""); break;

            }
        }

        public async static Task<Boolean> IsFlyingAsync()
        {
            var res = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).GetIsFlyingAsync();
            return res.value.Value.value;
        }

        public async static Task SetAircraftLimitation()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).SetDistanceLimitEnabledAsync(new BoolMsg() { value = true });
                IsResultReceivedEvent("SetDistanceLimitEnabled =true Result: " + res.ToString());
                var res1 = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).SetDistanceLimitAsync(new IntMsg() { value = 15 });
                IsResultReceivedEvent("SetDistanceLimit = 15 Result " + res1.ToString());
                var res2 = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).SetHeightLimitAsync(new IntMsg() { value = 20 });
                IsResultReceivedEvent("SetHeightLimit = 20 Result: " + res2.ToString()) ;
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("SetAirCraftLimitation Error", ex.ToString());
            }
        }

        public async static Task SetAircraftFailSafeAction()
        {
            try
            {
                FCFailsafeActionMsg fCFailsafeActionMsg;
                fCFailsafeActionMsg.value = FCFailsafeAction.LANDING;
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightControllerHandler(0, 0).SetFailsafeActionAsync(fCFailsafeActionMsg);
                IsResultReceivedEvent("SetFailsafeAction = Landing Result: " + res.ToString());

            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("SetAircraftFailSafeAction Error", ex.ToString());
            }

        }

        public async static Task EnableLandingProtection()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightAssistantHandler(0, 0).SetLandingProtectionEnabledAsync(new BoolMsg() { value = false });
                IsResultReceivedEvent("EnableLandingProtection = false Result: " +res.ToString());
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("EnableLandingProtection Error", ex.ToString());
            }
        }

        public async static Task EnableObstacleAvoidance()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightAssistantHandler(0, 0).SetObstacleAvoidanceEnabledAsync(new BoolMsg() { value = true });
                IsResultReceivedEvent("EnableObstacleAvoidance = true Result: " + res.ToString());
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("EnableObstacleAvoidance Error", ex.ToString());
            }
        }
        public async static Task EnableUpwardsAvoidance()
        {
            try
            {
                var res = await DJISDKManager.Instance.ComponentManager.GetFlightAssistantHandler(0, 0).SetUpwardsAvoidanceEnableAsync(new BoolMsg() { value = true });
                IsResultReceivedEvent("EnableUpwardsAvoidance Result: " + res.ToString());
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("EnableUpwardsAvoidance Error", ex.ToString());
            }
        }

    }
}
