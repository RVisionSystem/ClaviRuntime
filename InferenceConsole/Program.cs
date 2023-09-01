using ClaviRuntime;
using OpenCvSharp;
using System;
using System.Drawing;
using System.Text.Json;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics.Metrics;

class Program
{
    static void Main(string[] args)
    {
        string imagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\anomaly\\dataset\\OK\\OK_0000_20230404071725.jpg";
        string modelPath = "C:\\Users\\Beck\\Downloads\\anomalymodel(2)\\end2end.onnx";

        var anomalyDetection = new Anomaly();
        anomalyDetection.InitializeModel(modelPath);
        Bitmap resultImage = anomalyDetection.Process(imagePath);
        List<AnomalyResults>? results = anomalyDetection.resultsList;

        foreach (var r in results)
        {
            //Console.WriteLine("Output for {0}", r.Heatmap);
            Console.WriteLine(r.Score);
            Bitmap res = r.Heatmap;
            res.Save("C:\\Users\\Beck\\Desktop\\original.jpg");
        }
        //resultImage.Save("C:\\Users\\Beck\\Desktop\\original.jpg");

        Console.WriteLine("Done");
    }
}