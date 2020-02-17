using System;
using System.Collections.Concurrent;
using System.Threading;

namespace DTP.Utils
{
    class ObjectTracking
    {
        private int frame_width = 0;
        private int frame_height = 0;
        private int[] first_bounding_box_value;

        private int[] desiredROICenterCorr;
        private readonly double acceptableError = 0.30;
        private readonly float defaultJoyStickValue = 0.15F;

        private Boolean isIssueControlSignalTaskEnabled = false;
        private Boolean isStopThread = false;
        private ConcurrentQueue<int[]> boundingBoxQueue = new ConcurrentQueue<int[]>();

        public int Frame_width { get => frame_width; set => frame_width = value; }
        public int Frame_height { get => frame_height; set => frame_height = value; }
        public ConcurrentQueue<int[]> BoundingBoxQueue { get => boundingBoxQueue; set => boundingBoxQueue = value; }
        public bool IsIssueControlSignalTaskEnabled { get => isIssueControlSignalTaskEnabled; set => isIssueControlSignalTaskEnabled = value; }
        public bool IsStopThread { get => isStopThread; set => isStopThread = value; }


        private void IssueControlSignalTask()
        {
            while (!isStopThread)
            {
                while (isIssueControlSignalTaskEnabled)
                {
                    if (BoundingBoxQueue.IsEmpty)
                    {
                        Thread.Sleep(20);
                    }
                    else
                    {
                        int[] TmpBboxValues;
                        if (BoundingBoxQueue.TryDequeue(out TmpBboxValues))
                        {
                            ComputeNewThrottleAndAttitude(TmpBboxValues);
                            Controller.UpdateJoyStickValue();
                        }
                        else
                        {
                            Thread.Sleep(20);
                        }
                    }
                }
            }
        }

        public void ComputeNewThrottleAndAttitude(int [] TmpBboxValues)
        {
            if (first_bounding_box_value == null)
            {
                SetDesiredROISizeAndPosition(TmpBboxValues);
            }
            else
            {
                Controller.curr_yaw = CalcYawValue(CalcROICenterCorr(TmpBboxValues));
                if (Controller.curr_yaw == 0)
                {
                    Controller.curr_pitch = CalcPitchValue(CalcROIArea(TmpBboxValues));

                    if (Controller.curr_pitch == 0)
                    {
                        //Controller.curr_throttle = CalcThrottleValue(CalcROICenterCorr(TmpBboxValues));
                    }
                    else
                    {
                        //Controller.curr_throttle = 0;
                    }
                }
                else
                {
                    Controller.curr_pitch = 0;
                }
            }
        }

        public void StartIssueControlSignalTask()
        {
            var IssueControlThread = new Thread(new ThreadStart(IssueControlSignalTask));
            IssueControlThread.Start();
        }


        private void SetDesiredROISizeAndPosition(int[] TmpBboxValues)
        {

            //Save the desired ROI size and ROI position (Center Corr of ROI)
            first_bounding_box_value = TmpBboxValues;
            desiredROICenterCorr = new int[] { frame_width / 2, frame_height / 2 };

        }

        private float CalcPitchValue(int curr_ROI_area)
        {
            // devive the default object size within the tracking period from the selected roi
            int default_ROI_Area = CalcROIArea(first_bounding_box_value);
            double ChangeOfPercentage = CalcChangeOfPercentage(curr_ROI_area, default_ROI_Area);
            if (Math.Abs(ChangeOfPercentage) > acceptableError)
            {
                if (ChangeOfPercentage > 0)
                {
                    return -defaultJoyStickValue;
                }
                else
                {
                    return defaultJoyStickValue;
                }
            }
            else
            {
                return 0;
            }
        }

        private float CalcYawValue(int [] curr_ROI_CenterCorr)
        {
            int [] default_ROI_CenterCorr = CalcROICenterCorr(first_bounding_box_value);
            double ChangeOfPercentage = CalcChangeOfPercentage(curr_ROI_CenterCorr[0], default_ROI_CenterCorr[0]);

            if (Math.Abs(ChangeOfPercentage) > acceptableError)
            {
                if  (ChangeOfPercentage > 0)
                {
                    return +defaultJoyStickValue;
                }
                else
                {
                    return -defaultJoyStickValue;
                }
            }
            else
            {
                return 0;
            }
        }

        private float CalcThrottleValue(int[] curr_ROI_CenterCorr)
        {
            int[] default_ROI_CenterCorr = CalcROICenterCorr(first_bounding_box_value);
            double ChangeOfPercentage = CalcChangeOfPercentage(curr_ROI_CenterCorr[1], default_ROI_CenterCorr[1]);

            if (Math.Abs(ChangeOfPercentage) > acceptableError)
            {
                if (ChangeOfPercentage > 0)
                {
                    return -defaultJoyStickValue;
                }
                else
                {
                    return defaultJoyStickValue;
                }
            }
            else
            {
                return 0;
            }
        }

        private int[] CalcROICenterCorr(int[] bounding_box_value)
        {
            return new int[] { bounding_box_value[0] + bounding_box_value[2] / 2, bounding_box_value[1] + bounding_box_value[3] / 2 };
        }

        private int CalcROIArea(int[] bounding_box_value)
        {
            return bounding_box_value[2] * bounding_box_value[3];
        }

        private double CalcChangeOfPercentage(int new_value, int old_value)
        {
            return ((double)new_value - (double)old_value) / (double)old_value;
        }
    }


}
