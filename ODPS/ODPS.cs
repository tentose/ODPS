using System;
using System.Collections.Generic;
//using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;

namespace ODPS
{
    internal class ODPS
    {
        ScreenCapture capture = new ScreenCapture();
        ChatLineProcessor processor = new ChatLineProcessor();
        Timer mainTimer;
        Timer dpsCalcTimer;
        Size windowSize = new Size(2560, 1440);
        Rect windowRoi = new Rect(0, 0, 2560, 1440);

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

        private TimeSpan ClearDamageTimeout = TimeSpan.FromSeconds(10);
        private TimeSpan DamageWindow = TimeSpan.FromSeconds(30);
        private List<(int damage, DateTime time)> damageDealt = new List<(int damage, DateTime time)>();

        enum MarkerSearchStrategy
        {
            RedChannel,
            ToGray,
        };

        public ODPS()
        {
            // Load markers
            foreach (var kv in markerFilePaths)
            {
                if (!markerImages.ContainsKey(kv.Key))
                {
                    var tmp = Cv2.ImRead(GetPathForMarker(kv.Key));
                    Mat markerGray = new Mat();
                    Cv2.CvtColor(tmp, markerGray, ColorConversionCodes.BGR2GRAY);
                    markerImages[kv.Key] = markerGray;
                }
            }

            mainTimer = new Timer(TimerTick, null, 200, 200);
            dpsCalcTimer = new Timer(DpsCalcTimerTick, null, 1000, 1000);
        }

        private List<ChatLineContent> lastChatContent = new List<ChatLineContent>();

        public void DpsCalcTimerTick(Object? stateInfo)
        {
            int totalDamage = 0;
            int oldDamageIndex = -1;
            DateTime now = DateTime.Now;
            for (int i = 0; i < damageDealt.Count; i++)
            {
                if (damageDealt[i].time + DamageWindow < now)
                {
                    oldDamageIndex = i;
                }
                else
                {
                    totalDamage += damageDealt[i].damage;
                }
            }

            if (damageDealt.Count > 0 && damageDealt[damageDealt.Count - 1].time + ClearDamageTimeout < now)
            {
                damageDealt.Clear();
            }
            else if (oldDamageIndex >= 0)
            {
                damageDealt.RemoveRange(0, oldDamageIndex + 1);
            }
            
            if (damageDealt.Count > 0)
            {
                TimeSpan damageDuration = now - damageDealt[0].time;
                var seconds = damageDuration.TotalSeconds;
                Console.WriteLine($"{totalDamage / seconds}: {totalDamage} over {seconds} seconds");
            }
        }

        public void TimerTick(Object? stateInfo)
        {
            var bmp = capture.Capture(windowSize, windowRoi);
            if (bmp != null)
            {
                var bits = bmp.LockBits(new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

                try
                {
                    var original = new Mat(bits.Height, bits.Width, MatType.CV_8UC3, bits.Scan0, bits.Stride);

                    if (original.Width == windowSize.Width && original.Height == windowSize.Height)
                    {
                        // need to find markers to size things properly
                        var topRightMarkerPos = LookForMarker(original, MarkerType.ChatUpperRight, MarkerType.None, MarkerSearchStrategy.ToGray, 0.1);
                        var bottomLeftMarkerPos = LookForMarker(original, MarkerType.ChatBottomLeft, MarkerType.None, MarkerSearchStrategy.ToGray, 0.1);

                        if (topRightMarkerPos == null || bottomLeftMarkerPos == null)
                        {
                            Console.WriteLine("Cannot find markers");
                            return;
                        }

                        var left = bottomLeftMarkerPos.Value.X + 13;
                        var top = topRightMarkerPos.Value.Y + 30;
                        var bottom = bottomLeftMarkerPos.Value.Y - 5;
                        var right = topRightMarkerPos.Value.X + markerImages[MarkerType.ChatUpperRight].Width;
                        windowRoi = new Rect(left, top, right - left, bottom - top);

                        return;
                    }

                    var result = processor.ProcessChatScreen(original);

                    int newEntryCount = TryMatchChatContent(lastChatContent, result);
                    int indexOfFirstNewItem = result.Count - newEntryCount;
                    for (int i = indexOfFirstNewItem; i < result.Count; i++)
                    {
                        damageDealt.Add((result[i].Value, DateTime.Now));
                        Console.WriteLine($"{result[i].Type}: {result[i].Value}");
                    }

                    if (indexOfFirstNewItem < 2)
                    {
                        Console.WriteLine($"Large rewrite. indexOfFirstNewItem:{indexOfFirstNewItem}");
                        Console.WriteLine("New-----------------------------------");
                        for (int i = 0; i < result.Count; i++)
                        {
                            Console.WriteLine($"{result[i].Type}: {result[i].Value}");
                        }
                        Console.WriteLine("Old-----------------------------------");
                        for (int i = 0; i < lastChatContent.Count; i++)
                        {
                            Console.WriteLine($"{lastChatContent[i].Type}: {lastChatContent[i].Value}");
                        }
                        Console.WriteLine("End-----------------------------------");
                    }

                    lastChatContent = result;
                }
                finally
                {
                    bmp.UnlockBits(bits);
                }
            }
            //CvInvoke.WaitKey(50);
        }

        private int TryMatchChatContent(List<ChatLineContent> oldContent, List<ChatLineContent> newContent)
        {
            // find longest common subsequence and count out the extra entries in the new content
            int oldLength = oldContent.Count;
            int newLength = newContent.Count;
            int[,] path = new int[oldLength + 1, newLength + 1];

            for (int i = 0; i <= oldLength; i++)
            { 
                path[i, 0] = 0;
            }

            for (int j = 0; j <= newLength; j++)
            {
                path[0, j] = 0;
            }

            for (int i = 1; i <= oldLength; i++)
            {
                for (int j = 1; j <= newLength; j++)
                {
                    if (oldContent[i - 1] == newContent[j - 1])
                    {
                        path[i, j] = path[i - 1, j - 1] + 1;
                    }
                    else
                    {
                        path[i, j] = Math.Max(path[i, j - 1], path[i - 1, j]);
                    }
                }
            }

            /*
            for (int i = 1; i <= oldLength; i++)
            {
                for (int j = 1; j <= newLength; j++)
                {
                    Console.Write($"{path[i, j],3}");
                }
                Console.WriteLine();
            }
            */

            // # of new entries is on the last row
            int val = path[oldLength, newLength];
            int newEntries = 0;
            for (int j = newLength - 1; j >= 0; j--)
            {
                if (path[oldLength, j] == val)
                {
                    newEntries++;
                }
                else
                {
                    break;
                }
            }
            return newEntries;
        }

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
    }
}
