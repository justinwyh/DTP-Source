using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;

namespace DTP.Utils
{
    class APPConfig
    {
        ApplicationDataContainer localSettings = ApplicationData.Current.LocalSettings;

        public readonly static APPConfig instance = new APPConfig();

        public string getConfigProperties(String PropertyName)
        {
            return localSettings.Values[PropertyName] as String;
        }

        public void setConfigProperties(string PropertyName, string value)
        {
            localSettings.Values[PropertyName] = value;
        }
    }
}
