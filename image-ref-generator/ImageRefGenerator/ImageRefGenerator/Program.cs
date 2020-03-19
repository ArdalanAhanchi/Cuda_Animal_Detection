using System;
using System.IO;

namespace ImageRefGenerator
{
    class Program
    {
        const string OPEN_IMAGES_PATH = @"..\..\..\..\..\..\images\open-images";

        static void WriteDetailToFile(string fileName, string labelSourcePath, string labelImagePath, bool isWindows)
        {
            string lineAddition = labelSourcePath + " " + labelImagePath;
            lineAddition = lineAddition.Replace("C:\\dev\\UWB\\CSS535\\Project\\Cuda_Animal_Detection\\", "..\\");
            using (StreamWriter sw = File.AppendText(fileName))
            {
                string textToWrite = lineAddition;
                if (!isWindows)
                {
                    textToWrite = textToWrite.Replace('\\', '/');
                }
                sw.WriteLine(textToWrite);
            }
        }

        static void CreateOIResourceFile(string pathToDir)
        {
            if (Directory.Exists(pathToDir) && Directory.Exists(Path.Combine(pathToDir, "Label")))
            {
                DirectoryInfo di = new DirectoryInfo(pathToDir);
                string windowsResourceFilePath = Path.Combine(OPEN_IMAGES_PATH, di.Name + "_oi_resource.windows.txt");
                string linuxResourceFilePath = Path.Combine(OPEN_IMAGES_PATH, di.Name + "_oi_resource.linux.txt");
                
                try
                {
                    if (File.Exists(windowsResourceFilePath))
                    {
                        File.Delete(windowsResourceFilePath);
                    }

                    if (File.Exists(linuxResourceFilePath))
                    {
                        File.Delete(linuxResourceFilePath);
                    }
                }
                catch(Exception ex)
                {
                    Console.WriteLine("Error occurred trying to delete existing resource file: " + ex.Message);
                }
                
                string[] labelFiles = Directory.GetFiles(Path.Combine(pathToDir, "Label"));
                if (labelFiles != null && labelFiles.Length > 0)
                {
                    foreach(string labelFile in labelFiles)
                    {
                        string fileName = Path.GetFileName(labelFile).Replace(".txt", string.Empty);
                        WriteDetailToFile(windowsResourceFilePath, Path.GetFullPath(labelFile), Path.Combine(pathToDir, fileName + ".jpg"), true);
                        WriteDetailToFile(linuxResourceFilePath, Path.GetFullPath(labelFile), Path.Combine(pathToDir, fileName + ".jpg"), false);
                    }
                }
            }
        }

        static void Main(string[] args)
        {
            
            string fullOIPath = Path.GetFullPath(OPEN_IMAGES_PATH);

            string[] oiSubDirs = Directory.GetDirectories(fullOIPath);

            foreach(string subDir in oiSubDirs)
            {
                CreateOIResourceFile(subDir);
            }
        }
    }
}
