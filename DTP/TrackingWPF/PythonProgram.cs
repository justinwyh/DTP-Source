using System;
using System.Diagnostics;
using System.Threading;

namespace TrackingWPF
{
    class PythonProgram
    {
        public delegate void TBOutputReceivedHandler(string s);
        public event TBOutputReceivedHandler tbORHandler;
        private string programName;
        private string programPath;

        public PythonProgram(string programName, string programPath)
        {
            this.programName = programName;
            this.programPath = programPath;
        }

        public void executePythonProgram(string arguments)
        {

            var thread = new Thread(new ThreadStart(() =>
            {
                try
                {
                    using (Process p = new Process())
                    {
                        p.StartInfo.FileName = programPath;
                        p.StartInfo.Arguments = arguments;
                        p.StartInfo.UseShellExecute = false;
                        p.StartInfo.RedirectStandardOutput = true;
                        p.StartInfo.RedirectStandardError = true;
                        p.StartInfo.CreateNoWindow = true;
                        p.OutputDataReceived += new DataReceivedEventHandler(PythonOutputStreamReceivedHandler);
                        p.Start();
                        p.BeginOutputReadLine();
                        tbORHandler(programName + " - Start");
                        p.WaitForExit();
                        p.Close();
                        tbORHandler(programName + " - End");
                    }
                }
                catch (Exception ex)
                {
                    tbORHandler(ex.ToString());
                }

            }));
            thread.Start();
        }

        private void PythonOutputStreamReceivedHandler(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null)
            {
                tbORHandler(e.Data.ToString());
            }
        }
    }
}
