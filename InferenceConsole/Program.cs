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
        // Anomaly
        /*        string imagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\anomaly\\dataset\\No-crop\\test03.png";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\anomaly\\model\\anoCLS.onnx";

                var anomaly = new Anomaly();
                anomaly.InitializeModel(modelPath);
                anomaly.Process(imagePath);
                List<AnomalyResults>? results = anomaly.resultsList;

                foreach (var r in results)
                {
                    Console.WriteLine("This image has a " +  r.Score + "% abnormality.");
                    Bitmap res = r.Heatmap;
                    res.Save("C:\\Users\\Beck\\Desktop\\original.jpg");
                }

                Console.WriteLine("Done");*/

        //Instance Segmentation
        /*        string imagePath = "C:\\Users\\Beck\\Pictures\\Original_Dataset\\Sample_instance\\IMG-0946.jpg";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\instance\\model\\Instance_Segment.onnx";

                var instanceSegmentation = new InstanceSegmentation();
                instanceSegmentation.InitializeModel(modelPath);
                instanceSegmentation.Process(imagePath);
                Bitmap resultImage = instanceSegmentation.imageResult;
                resultImage.Save("C:\\Users\\Beck\\Desktop\\IMG-0946_result.jpg");
                List<InstanceSegmentationResults> results = instanceSegmentation.resultsList;
                if (results.Count != 0)
                {
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output for {0}", r.Id);
                        Console.WriteLine(r.Name);
                        Console.WriteLine(r.Score);
                    }
                }

                Console.WriteLine("Done");*/

        //Image Classification
        /*        string imagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\classification\\tire\\image\\ASTM-F2870\\ASTM-F2870_30.JPG";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\classification\\tire\\model\\20230905.onnx";
                Mat inputImg = Cv2.ImRead(imagePath);

                var classify = new ImageClassification();
                classify.InitializeModel(modelPath);
                classify.Process(imagePath);
                List<ImageClassificationResults> results = classify.resultsList;
                if (results.Count != 0)
                {
                    foreach (var r in results)
                    {
                        Console.WriteLine(r.Name);
                        Console.WriteLine(r.Score);
                    }
                }*/

        //Object Detection
        string imagePath = "C:\\Users\\Beck\\Pictures\\Original_Dataset\\Sample_Obj\\IMG-0970.jpg";
        string modelPath = "C:\\Users\\Beck\\Downloads\\object detection\\end2end.onnx";

        var obj = new ObjectDetection();
        obj.InitializeModel(modelPath);
        obj.Process(imagePath);
        var resultImage = obj.imageResult;
        List<ObjectDetectionResults>? results = obj.resultList;
        if (results != null)
        {
            foreach (var r in results)
            {
                Console.WriteLine("Output for {0}", r.Index);
                Console.WriteLine(r.Label);
                Console.WriteLine(r.Score);
                Console.WriteLine(r.Box[0].ToString() + " | " + r.Box[1].ToString() + " | " + r.Box[2].ToString() + " | " + r.Box[3].ToString());
            }
        }

        resultImage.Save("C:\\Users\\Beck\\Desktop\\results.jpg");
        Console.WriteLine("Done");

        //Semantic Segmentation
        /*        string imagePath = "C:\\Users\\Beck\\Pictures\\Dataset\\RankI\\Rank1_01.JPG";
                string modelPath = "C:\\Users\\Beck\\Downloads\\semantic\\end2end.onnx";

                var semanticSegmentation = new SemanticSegmentation();
                semanticSegmentation.InitializeModel(modelPath);
                Bitmap resultImage = semanticSegmentation.Process(imagePath);
                List<SemanticSegmentationResults>? results = semanticSegmentation.resultsList;
                foreach (var r in results)
                {
                    Bitmap res = r.Mask;
                }
                resultImage.Save("C:\\Users\\Beck\\Desktop\\original.jpg");

                Console.WriteLine("Done");*/


    }
}