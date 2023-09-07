using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using OpenCvSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats;
using System.Drawing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace ClaviRuntime
{
    public class ImageClassification : IDisposable
    {
        private InferenceSession? sess;
        public List<ImageClassificationResults>? resultsList;

        public void InitializeModel(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }

        public void Process(string imagePath, float threshold = 0.7F)
        {
            resultsList = new List<ImageClassificationResults>();
            Mat image = Cv2.ImRead(imagePath);
            try
            {
                // Setup inputs and outputs
                var inputMeta = sess.InputMetadata;
                var inputName = inputMeta.Keys.ToArray()[0];
                var meta = inputMeta[inputName];
                int[] dims = meta.Dimensions;
                var customMeta = sess.ModelMetadata.CustomMetadataMap;
                string lab = customMeta.ToArray()[0].Value;
                var lab_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(lab);

                OpenCvSharp.Size imgSize = new OpenCvSharp.Size(dims[3], dims[2]);

                Mat imageFloat = image.Resize(imgSize);
                imageFloat.ConvertTo(imageFloat, MatType.CV_32FC1);
                var input = new DenseTensor<float>(MatToList(imageFloat), new[] { dims[0], dims[1], dims[2], dims[3] });

/*                using Image<Rgb24> imageSix = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath, out IImageFormat format);
                using Stream imageStream = new MemoryStream();
                imageSix.Mutate(x =>
                {
                    x.Resize(new ResizeOptions 
                    {
                        Size = new SixLabors.ImageSharp.Size(dims[3], dims[2]),
                        Mode = ResizeMode.Crop
                    }); 
                });
                imageSix.Save(imageStream, format);

                Tensor<float> input = new DenseTensor<float>(new[] { dims[0], dims[1], dims[2], dims[3] });
                var mean = new[] { 123.675, 116.28, 103.53 };
                var stddev = new[] { 58.395, 57.12, 57.375 };
                imageSix.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (int x = 0; x < accessor.Width; x++)
                        {
                            input[0, 0, y, x] = (float)((pixelSpan[x].R - mean[0]) / stddev[0]) / 255f;
                            input[0, 1, y, x] = (float)((pixelSpan[x].G - mean[1]) / stddev[1]) / 255f;
                            input[0, 2, y, x] = (float)((pixelSpan[x].B - mean[2]) / stddev[2]) / 255f;
                        }

                    }
                });*/

                var inputs = new List<NamedOnnxValue>{NamedOnnxValue.CreateFromTensor(inputName, input)};
                using (var results = sess.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                    float maxValue = pred_value.Max();
                    int maxIndex = pred_value.ToList().IndexOf(maxValue);

                    resultsList.Add(new ImageClassificationResults(lab_dict[maxIndex.ToString()], pred_value[maxIndex]));
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
        public void Dispose()
        {
            sess?.Dispose();
            sess = null;
        }

    }
}
