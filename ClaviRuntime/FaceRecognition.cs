using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class FaceRecognition
    {
        private InferenceSession? sess;
        public Bitmap? imageResult;
        public List<ObjectDetectionResults>? resultList;

        public void InitializeModel(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }

        public void Process(string imagePath1, string imagePath2)
        {
            try
            {
                var inputMeta = sess.InputMetadata;
                var inputName1 = inputMeta.Keys.ToArray()[0];
                var inputName2 = inputMeta.Keys.ToArray()[1];
                var class_name = sess.ModelMetadata.CustomMetadataMap;
                int[] dims = inputMeta[inputName1].Dimensions;
                //var customMeta = sess.ModelMetadata.CustomMetadataMap;
                //string lab = customMeta.ToArray()[0].Value;
                //var lab_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(lab);

                Mat image1 = Cv2.ImRead(imagePath1, ImreadModes.Grayscale);
                Mat image2 = Cv2.ImRead(imagePath2, ImreadModes.Grayscale);

                float nmsThresh = 0.4f;
                // inputW = dims[3]; inputH = dims[2];
                OpenCvSharp.Size imgSize = new OpenCvSharp.Size(dims[3], dims[2]);

                Mat imageFloat1 = image1.Resize(imgSize);
                imageFloat1.ConvertTo(imageFloat1, MatType.CV_32FC1, 1 / 255.0f);

                Mat imageFloat2 = image2.Resize(imgSize);
                imageFloat2.ConvertTo(imageFloat2, MatType.CV_32FC1, 1 / 255.0f);

                var input1 = new DenseTensor<float>(MatToList(imageFloat1), new[] { dims[0], dims[1], dims[2], dims[3] });
                var input2 = new DenseTensor<float>(MatToList(imageFloat2), new[] { dims[0], dims[1], dims[2], dims[3] });

                //var labelList = GetLabel(class_name).ToArray();
                //var pallete = GenPalette(lab_dict.Count);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName1, input1),
                    NamedOnnxValue.CreateFromTensor(inputName2, input2)
                };

                using (var results = sess.Run(inputs))
                {
                    //Postprocessing
                    var resultsArray = results.ToArray();
                    var face1 = resultsArray[0].AsEnumerable<float>().ToArray();
                    var face2 = resultsArray[1].AsEnumerable<float>().ToArray();
                    //var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                    //var label_value = resultsArray[1].AsEnumerable<Int64>().ToArray();
                    float dissim = CalculateEuclideanDistance(face1, face2);
                    Console.WriteLine("Dissimilarity: " + dissim);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Model not initialized");
                Console.WriteLine(e.Message);
            }

        }

        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }
        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[chn * ih * iw];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }

        public static float CalculateEuclideanDistance(float[] point1, float[] point2)
        {
            if (point1.Length != point2.Length)
            {
                throw new ArgumentException("Both input points must have the same dimension.");
            }

            float sum = 0.0f;

            for (int i = 0; i < point1.Length; i++)
            {
                float diff = (point1[i] - point2[i]);
                sum += diff * diff;
            }

            return (float)Math.Sqrt(sum);
        }

        public static void dissimilarity(float[] point1, float[] point2)
        {
            float distance = CalculateEuclideanDistance(point1, point2);
        }
    }
}
