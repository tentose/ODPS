using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace ODPSCore
{
    public enum ChatLineType
    {
        Unknown,
        Hit,
        CriticalHit,
    }

    public record ChatLineContent(ChatLineType Type, int Value);

    public class ChatLineProcessor
    {
        private const double TEMPLATE_MATCH_THREASHOLD = 0.05;

        private int LineStart = -1;

        private enum AlphabetKeywordType
        {
            You,
            Critically,
            Hit,
            Using,
        };

        private const string ALPHABET_PATH = @".\alphabet\";
        Dictionary<int, (Mat img, Mat? mask)> Numbers = new Dictionary<int, (Mat, Mat?)>();
        Dictionary<string, (Mat img, Mat? mask)> Keywords = new Dictionary<string, (Mat, Mat?)>();
        private void LoadAlphabet()
        {
            var files = Directory.EnumerateFiles(ALPHABET_PATH, "*.png");
            var loadSingle = (string filePath) =>
            {
                string maskPath = filePath.Replace(".png", ".mask.png");
                Mat? mask = null;
                if (File.Exists(maskPath))
                {
                    mask = new Mat(maskPath, ImreadModes.Grayscale);
                }

                var gray = new Mat(filePath).CvtColor(ColorConversionCodes.BGR2HSV).Split()[2];

                return (gray, mask);
            };

            foreach (var filePath in files)
            {
                string fileName = Path.GetFileNameWithoutExtension(filePath);

                // skip masks
                if (fileName.EndsWith(".mask"))
                {
                    continue;
                }
                
                if (fileName.Length == 1 && fileName[0] >= '0' && fileName[0] <= '9')
                {
                    // is number
                    int value = fileName[0] - '0';
                    var images = loadSingle(filePath);
                    Numbers[value] = (images.gray, images.mask);
                }

                if (fileName.Length > 1 && !fileName.StartsWith("lower-") && fileName != "colon")
                {
                    // is keyword
                    var images = loadSingle(filePath);
                    Keywords[fileName] = (images.gray, images.mask);
                }
            }
        }

        public ChatLineProcessor()
        {
            LoadAlphabet();
        }

        public List<ChatLineContent> ProcessChatScreen(Mat img)
        {
            List<ChatLineContent> resultLines = new List<ChatLineContent>();
            var gray = img.CvtColor(ColorConversionCodes.BGR2HSV).Split()[2];

            Mat outputMask = Mat.Zeros(gray.Height + 2, gray.Width + 2, MatType.CV_8UC1);
            Cv2.FloodFill(gray, new Point(0, 0), new Scalar(0), out Rect rect, new Scalar(20), new Scalar(20), FloodFillFlags.Link8);

            Mat row = gray.Reduce(ReduceDimension.Column, ReduceTypes.Max, -1);           

            var output = gray;

            //Matrix<byte> rowFiltered = new Matrix<byte>(img.Rows, 1);
            //CvInvoke.Threshold(row, rowFiltered, 90, 255, Emgu.CV.CvEnum.ThresholdType.BinaryInv);

            //Cv2.ImShow("output", output);
            //Cv2.WaitKey();

            int lastPicBottom = 0;
            for (int i = 0; i < row.Rows; i++)
            {
                var height = i - lastPicBottom;
                if (row.At<byte>(i, 0) < 110)
                {
                    if (height > 15)
                    {
                        var line = gray[new Rect(0, lastPicBottom, img.Width, height)];
                        output.Line(new Point(0, i), new Point(img.Width, i), new Scalar(255), 1);

                        resultLines.Add(ProcessChatLine(line));
                    }
                    lastPicBottom = i;
                }
            }

            //CvInvoke.Imshow("input", output);
            //CvInvoke.WaitKey(50);

            

            return resultLines;
        }

        private Point? FindTemplate(Mat img, Mat template, Mat? mask)
        {
            Mat result = new Mat();
            if (mask == null)
            {
                Cv2.MatchTemplate(img, template, result, TemplateMatchModes.SqDiffNormed);
            }
            else
            {
                Cv2.MatchTemplate(img, template, result, TemplateMatchModes.SqDiffNormed, mask);
            }

            Cv2.MinMaxLoc(result, out double minVal, out double maxVal, out Point minLoc, out Point maxLoc);

            if (minVal < TEMPLATE_MATCH_THREASHOLD)
            {
                return new Point(minLoc.X, minLoc.Y);
            }

            return null;
        }

        private List<Point> FindTemplates(Mat img, Mat template, Mat? mask)
        {
            List<Point> matches = new List<Point>();

            Mat result = new Mat();
            if (mask == null)
            {
                Cv2.MatchTemplate(img, template, result, TemplateMatchModes.SqDiffNormed);
            }
            else
            {
                Cv2.MatchTemplate(img, template, result, TemplateMatchModes.SqDiffNormed, mask);
            }

            var minmaxMask = Mat.Ones(result.Rows, result.Cols, MatType.CV_8UC1);
            while (true)
            {
                Cv2.MinMaxLoc(result, out double minVal, out double maxVal, out Point minLoc, out Point maxLoc, minmaxMask);

                if (minVal < TEMPLATE_MATCH_THREASHOLD)
                {
                    int x = minLoc.X;
                    int y = minLoc.Y;
                    int w = template.Width;
                    int h = template.Height;
                    Rect rect = new Rect(x, y, w, h);

                    Cv2.Rectangle(minmaxMask, rect, new Scalar(0), -1);

                    matches.Add(new Point(x, y));
                }
                else
                {
                    break;
                }
            }

            return matches;
        }

        private Mat GetSubRect(Mat img, int start, int end)
        {
            return img[new Rect(start, 0, end - start, img.Height)];
        }

        public ChatLineContent ProcessChatLine(Mat img)
        {
            List<(int x, string s)> matches = new List<(int x, string s)>();

            ChatLineType chatLineType = ChatLineType.Unknown;
            int chatLineValue = 0;

            var youTemplate = Keywords["You"];
            var hitTemplate = Keywords["hit"];
            var critTemplate = Keywords["critically"];

            // Find "You"
            int searchStart = LineStart >= 0 ? LineStart : 0;
            Point? youResult = FindTemplate(GetSubRect(img, searchStart, img.Width / 3), youTemplate.img, youTemplate.mask);
            if (youResult == null)
            {
                return new ChatLineContent(chatLineType, chatLineValue);
            }
            if (LineStart < 0)
            {
                LineStart = youResult.Value.X - 20;
                LineStart = LineStart < 0 ? 0 : LineStart;
            }
            int youEnd = searchStart + youResult.Value.X + youTemplate.img.Width;

            // Is this a hit?
            Point? hitResult = FindTemplate(GetSubRect(img, youEnd, youEnd + hitTemplate.img.Width + critTemplate.img.Width + 20), hitTemplate.img, hitTemplate.mask);
            if (hitResult != null)
            {
                int hitStart = youEnd + hitResult.Value.X;
                int hitEnd = hitStart + hitTemplate.img.Width;
                chatLineType = ChatLineType.Hit;

                // Is this a crit?
                if (hitStart - youEnd > 60)
                {
                    // assume if there is enough space between "You" and "hit" then it's a crit
                    chatLineType = ChatLineType.CriticalHit;
                    /*
                    Point? criticallyResult = FindTemplate(GetSubRect(img, youEnd, hitStart), critTemplate.img, critTemplate.mask);

                    if (criticallyResult != null)
                    {
                        chatLineType = ChatLineType.CriticalHit;
                    }
                    */
                }

                // If this is a hit or crit, find the value
                
                // Try to scope the search
                int forEnd = 0;
                int usingStart = img.Width;

                // Find "for"
                var forTemplate = Keywords["for"];
                Point? forResult = FindTemplate(GetSubRect(img, hitEnd, img.Width / 4 * 3), forTemplate.img, forTemplate.mask);
                if (forResult != null)
                {
                    forEnd = hitEnd + forResult.Value.X + forTemplate.img.Width;
                }

                // Find "using"
                var usingTemplate = Keywords["using"];
                Point? usingResult = FindTemplate(GetSubRect(img, forEnd, img.Width), usingTemplate.img, usingTemplate.mask);
                if (usingResult != null)
                {
                    usingStart = forEnd + usingResult.Value.X;
                }

                // Now find all the numbers
                List<(int x, int value)> allNumberMatches = new List<(int x, int value)>();
                var numberSearchImg = GetSubRect(img, forEnd, usingStart);
                foreach (var kv in Numbers)
                {
                    var numberMatches = FindTemplates(numberSearchImg, kv.Value.img, kv.Value.mask);
                    allNumberMatches.AddRange(numberMatches.Select(m => (m.X, kv.Key)));
                }
                allNumberMatches.Sort((a, b) => { return a.x.CompareTo(b.x); });

                int radix = 1;
                int total = 0;
                for (int i = allNumberMatches.Count - 1; i >= 0; i--)
                {
                    total += allNumberMatches[i].value * radix;
                    radix *= 10;
                }
                chatLineValue = total;
            }

            return new ChatLineContent(chatLineType, chatLineValue);
        }
    }
}
