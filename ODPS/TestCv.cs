using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace ODPS
{
    internal class TestCv
    {
        enum MarkerType
        {
            None,
            ChatUpperRight,
            ChatBottomLeft,
        };

        string markerFileParentPath = @".\markers\";
        Dictionary<MarkerType, string> markerFilePaths = new Dictionary<MarkerType, string>()
            {
                { MarkerType.ChatUpperRight, "ChatUpperRight.png" },
                { MarkerType.ChatBottomLeft, "ChatBottomLeft.png" },
            };
        private string GetPathForMarker(MarkerType type)
        {
            return markerFileParentPath + markerFilePaths[type];
        }
        Dictionary<MarkerType, Mat> markerImages = new Dictionary<MarkerType, Mat>();

        enum MarkerSearchStrategy
        {
            RedChannel,
            ToGray,
        };

        private Point? LookForMarker(Mat img, MarkerType markerType, MarkerType mask, MarkerSearchStrategy searchStrategy, double confidenceThreshold = 0.3)
        {
            Point? markerPosition = null;
            var markerImg = markerImages[markerType];

            Mat output = img.CvtColor(ColorConversionCodes.BGR2GRAY);

            Mat searchImg = null;
            if (searchStrategy == MarkerSearchStrategy.ToGray)
            {
                searchImg = img.CvtColor(ColorConversionCodes.BGR2GRAY);
            }
            else if (searchStrategy == MarkerSearchStrategy.RedChannel)
            {
                var channels = img.Split();

                var red = channels[2];
                searchImg = red;
            }

            Mat result = new Mat();
            if (mask == MarkerType.None)
            {
                Cv2.MatchTemplate(searchImg, markerImg, result, TemplateMatchModes.SqDiffNormed);
            }
            else
            {
                Cv2.MatchTemplate(searchImg, markerImg, result, TemplateMatchModes.SqDiffNormed, markerImages[mask]);
            }

            Cv2.MinMaxLoc(result, out double minVal, out double maxVal, out Point minLoc, out Point maxLoc);

            int x = minLoc.X;
            int y = minLoc.Y;
            int w = markerImg.Width;
            int h = markerImg.Height;
            Rect exclaimRect = new Rect(x, y, w, h);

            if (minVal < confidenceThreshold)
            {
                markerPosition = minLoc;
            }

            return markerPosition;
        }

        public void Exec()
        {
            // Load markers
            foreach (var kv in markerFilePaths)
            {
                if (!markerImages.ContainsKey(kv.Key))
                {
                    var tmp = new Mat(GetPathForMarker(kv.Key));
                    var markerGray = tmp.CvtColor(ColorConversionCodes.BGR2GRAY);
                    markerImages[kv.Key] = markerGray;
                }
            }


            var original = new Mat(@".\sample\Uneven2.png");
            var topRightMarkerPos = LookForMarker(original, MarkerType.ChatUpperRight, MarkerType.None, MarkerSearchStrategy.ToGray);
            var bottomLeftMarkerPos = LookForMarker(original, MarkerType.ChatBottomLeft, MarkerType.None, MarkerSearchStrategy.ToGray);

            if (topRightMarkerPos == null || bottomLeftMarkerPos == null)
            {
                Console.WriteLine("Cannot find markers");
                return;
            }

            var left = bottomLeftMarkerPos.Value.X + 13;
            var top = topRightMarkerPos.Value.Y + 30;
            var bottom = bottomLeftMarkerPos.Value.Y - 5;
            var right = topRightMarkerPos.Value.X + markerImages[MarkerType.ChatUpperRight].Width;
            var img = original[new Rect(left, top, right - left, bottom - top)];
            //var gray = img.Convert<Gray, byte>();
            var gray = img.CvtColor(ColorConversionCodes.BGR2HSV).Split()[2];

            ChatLineProcessor processor = new ChatLineProcessor();

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < 200; i++)
            {
                var res = processor.ProcessChatScreen(img);
            }
            Console.WriteLine(stopwatch.ElapsedMilliseconds);
            stopwatch.Stop();

            var result = processor.ProcessChatScreen(img);
            foreach (var line in result)
            {
                Console.WriteLine($"{line.Type}: {line.Value}");
            }

            //CvInvoke.WaitKey(0);
            Console.ReadLine();
        }

    }
}
