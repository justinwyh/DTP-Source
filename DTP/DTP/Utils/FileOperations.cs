using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.UI.Xaml.Controls;

namespace DTP.Utils
{
    class FileOperations
    {
        public static async Task HousekeepExtractedFramesFolder()
        {
            try
            {
                string ExtractedFrameFolderPath = APPConfig.instance.getConfigProperties("ExtractedFramesFolder");
                String outputDrivePath = APPConfig.instance.getConfigProperties("Drive");
                StorageFolder outputDrive = await StorageFolder.GetFolderFromPathAsync(outputDrivePath);
                if (await outputDrive.TryGetItemAsync(ExtractedFrameFolderPath) != null) { 

                IStorageItem ExtractedFramesFolder = await outputDrive.GetItemAsync(ExtractedFrameFolderPath);

                await ExtractedFramesFolder.DeleteAsync(StorageDeleteOption.PermanentDelete);
                }
                
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("Housekeep Folder Error", "Please go to Settings > Privacy > File system and enable the file access right of DTP."+ Environment.NewLine + ex.ToString());
            }
        }

        public static async Task CreateApplicationDataFolder()
        {
            try
            {
                StorageFolder outputLocation = await StorageFolder.GetFolderFromPathAsync(APPConfig.instance.getConfigProperties("Drive"));
                StorageFolder storageFolder = await outputLocation.CreateFolderAsync(APPConfig.instance.getConfigProperties("ExtractedFramesFolder"), CreationCollisionOption.OpenIfExists);
                
            }
            catch (Exception ex)
            {
                UIOperations.ShowContentDialog("Create Folder Error", "Please go to Settings > Privacy > File system and the enable file access right of DTP." + Environment.NewLine + ex.ToString());
            }
        }
    }


}
