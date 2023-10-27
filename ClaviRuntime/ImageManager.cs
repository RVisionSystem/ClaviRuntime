using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;

namespace ClaviRuntime
{
    public static class ImageManager
    {
        public static Mat Resize(Mat inputMat, int target_width, int target_height)
        {
            //Resize
            // Calculate the ratio of the new size to the original size
            float widthRatio = (float)target_width / (float)inputMat.Width;
            float heightRatio = (float)target_height / (float)inputMat.Height;
            float ratio = Math.Min(heightRatio, widthRatio);

            // Calculate the new width and height based on the ratio
            int newWidth = (int)(inputMat.Width * ratio);
            int newHeight = (int)(inputMat.Height * ratio);

            OpenCvSharp.Size newSize = new OpenCvSharp.Size(newWidth, newHeight); //Keep aspect ratio
            OpenCvSharp.Size No_aspect = new OpenCvSharp.Size(640, 640); //No aspect ratio

            //Resize with aspect ratio
            Mat resizedImage = inputMat.Resize(No_aspect);
            resizedImage.ConvertTo(resizedImage, MatType.CV_32FC3);
            //Calculate the top left bottom right padding
            OpenCvSharp.Size paddingSize = GetPaddingSize(new OpenCvSharp.Size(target_width, target_width), newSize);
            int TB = paddingSize.Height;
            int LR = paddingSize.Width;

            //Padding
            Mat padded = AddPadding(resizedImage, TB, TB, LR, LR, BorderTypes.Constant, new Scalar(114, 114, 114));

            return resizedImage;
        }

        public static Mat AddPadding(Mat src, int top, int bottom, int left, int right, BorderTypes borderTypes, Scalar value)
        {
            Mat dst = new Mat();
            Cv2.CopyMakeBorder(src, dst, top, bottom, left, right, borderTypes, value);
            return dst;
        }

        public static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }
        public unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
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

        private static OpenCvSharp.Size GetPaddingSize(OpenCvSharp.Size requiredSize, OpenCvSharp.Size newSize)
        {
            int TB = (requiredSize.Height - newSize.Height) / 2;
            int LR = (requiredSize.Width - newSize.Width) / 2;

            OpenCvSharp.Size paddingSize = new OpenCvSharp.Size(LR, TB);

            return paddingSize;
        }

        public static Image ToImageSharpImage(System.Drawing.Bitmap bitmap)
        {
            using (var memoryStream = new MemoryStream())
            {
                bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Jpeg);

                memoryStream.Seek(0, SeekOrigin.Begin);

                return Image.Load(memoryStream);
            }
        }

    }
}
