using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using OpenCvSharp.Flann;
using Newtonsoft.Json;
using System.Data;
using System.Runtime.Intrinsics.X86;

namespace ClaviRuntime
{
    public class FaceRecognition
    {
        private InferenceSession? sess;
        public Bitmap? imageResult;
        public List<FaceRecognitionResults>? resultList;

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
                Dictionary<string, float> errPair = new Dictionary<string, float>();
                var inputMeta = sess.InputMetadata;
                var inputName1 = inputMeta.Keys.ToArray()[0];
                var inputName2 = inputMeta.Keys.ToArray()[1];
                var class_name = sess.ModelMetadata.CustomMetadataMap;
                int[] dims = inputMeta[inputName1].Dimensions;
                //var customMeta = sess.ModelMetadata.CustomMetadataMap;
                //string lab = customMeta.ToArray()[0].Value;
                //var lab_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(lab);

                Mat imageTarget = Cv2.ImRead(imagePath2, ImreadModes.Grayscale);

                // inputW = dims[3]; inputH = dims[2];
                OpenCvSharp.Size imgSize = new OpenCvSharp.Size(dims[3], dims[2]);

                Mat imageTargetFloat = imageTarget.Resize(imgSize);
                imageTargetFloat.ConvertTo(imageTargetFloat, MatType.CV_32FC1, 1 / 255.0f);
                var input1 = new DenseTensor<float>(MatToList(imageTargetFloat), new[] { dims[0], dims[1], dims[2], dims[3] });

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName1, input1), //input same image
                    NamedOnnxValue.CreateFromTensor(inputName2, input1)
                };

                using (var results = sess.Run(inputs))
                {
                    //Postprocessing
                    var resultsArray = results.ToArray();
                    var targetFace = resultsArray[0].AsEnumerable<float>().ToArray();
                    //var face2 = resultsArray[1].AsEnumerable<float>().ToArray();
                    var result = CalculateAverageEuclideanDistance(targetFace);

                    // return the minimun error
                    string resName = GetKeyOfMinValue(result);
                    Console.WriteLine(resName);
                    //resultList.Add(new FaceRecognitionResults(resName));

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

        public static void dissimilarity(float[] pointDB, float[] pointTarget)
        {
            float distance = CalculateEuclideanDistance(pointDB, pointTarget);
        }

        public void CreateDB() // float[] inputFace
        {
            Dictionary<string, List<float[]>> dbFace = new Dictionary<string, List<float[]>>();
            
            var sess = new InferenceSession("C:\\Users\\Beck\\Model\\model-test-lib\\face\\model\\face-recog22_9.onnx");
            var inputMeta = sess.InputMetadata;
            var inputName1 = inputMeta.Keys.ToArray()[0];
            var inputName2 = inputMeta.Keys.ToArray()[1];
            var class_name = sess.ModelMetadata.CustomMetadataMap;
            int[] dims = inputMeta[inputName1].Dimensions;

            string peoplePath = "C:\\Users\\Beck\\Model\\model-test-lib\\face\\";
            foreach (string peopleFileName in Directory.GetFiles(peoplePath, "*.txt"))
            {
                // file name without extension
                string fileName = Path.GetFileNameWithoutExtension(peopleFileName);
                string ImagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\face\\" + fileName + "\\";
                int ind = 0;
                JObject sing = new JObject();
                foreach (string imageFileName in Directory.GetFiles(ImagePath, "*.jpg"))
                {
                    Mat image1 = Cv2.ImRead(imageFileName, ImreadModes.Grayscale);

                    OpenCvSharp.Size imgSize = new OpenCvSharp.Size(dims[3], dims[2]);

                    Mat imageFloat1 = image1.Resize(imgSize);
                    imageFloat1.ConvertTo(imageFloat1, MatType.CV_32FC1, 1/255.0f);

                    var input1 = new DenseTensor<float>(MatToList(imageFloat1), new[] { dims[0], dims[1], dims[2], dims[3] });

                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName1, input1), NamedOnnxValue.CreateFromTensor(inputName2, input1) };

                    using (var results = sess.Run(inputs))
                    {
                        JArray eachFace = new JArray();
                        var resultsArray = results.ToArray();
                        var face1 = resultsArray[0].AsEnumerable<float>().ToArray();
                        foreach (var face in face1)
                        {
                            eachFace.Add(face);
                        }
                        sing[ind.ToString()] = eachFace;
                        ind++;
                    }
                }
                JObject o = new JObject();
                o[fileName] = sing;
                string js = o.ToString();
                Console.WriteLine(js);
            }
        }

        public void ReadJSON()
        {
            string json = File.ReadAllText("C:\\Users\\Beck\\Model\\model-test-lib\\face\\10points.json");
            var faceDict = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, float[]>>>(json);
            foreach (var item in faceDict)
            {
                Console.WriteLine(item.Key);
                foreach (var item2 in item.Value)
                {
                    Console.WriteLine(item2.Key + " - " + item2.Value[0]);
                }
            }
        }

        public static Dictionary<string, float> CalculateAverageEuclideanDistance(float[] targetPnt)
        {

            Dictionary<string, float> errPair = new Dictionary<string, float>();
            string json = File.ReadAllText("C:\\Users\\Beck\\Model\\model-test-lib\\face\\10points.json");
            var faceDict = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, float[]>>>(json);
            foreach (var item in faceDict)
            {
                // average with the item.Value.Count()
                float numFaces = item.Value.Count();
                float sum = 0f;

                foreach (var item2 in item.Value)
                {
                    // Find Dissimilatlity
                    float a = CalculateEuclideanDistance(item2.Value, targetPnt);
                    sum += a;
                }
                errPair.Add(item.Key, (float)(sum/numFaces));
                //Console.WriteLine(item.Key + " - " + (float)(sum / numFaces));
            }
            return errPair;
        }

        static TKey GetKeyOfMinValue<TKey, TValue>(IDictionary<TKey, TValue> dictionary) where TValue : IComparable<TValue>
        {
            KeyValuePair<TKey, TValue> min = dictionary.First();
            foreach (KeyValuePair<TKey, TValue> pair in dictionary.Skip(1))
            {
                if (pair.Value.CompareTo(min.Value) < 0)
                {
                    min = pair;
                }
            }
            return min.Key;
        }

    }
}
