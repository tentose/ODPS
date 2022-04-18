using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;

namespace ODPS
{
    public enum ChatLineType
    {
        Unknown,
        Hit,
        CriticalHit,
    }

    public record ChatLineContent(ChatLineType Type, int Value);

    internal class ChatLineProcessor
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
        Dictionary<int, (Image<Gray, byte> img, Image<Gray, byte>? mask)> Numbers = new Dictionary<int, (Image<Gray, byte>, Image<Gray, byte>?)>();
        Dictionary<string, (Image<Gray, byte> img, Image<Gray, byte>? mask)> Keywords = new Dictionary<string, (Image<Gray, byte>, Image<Gray, byte>?)>();
        private void LoadAlphabet()
        {
            var files = Directory.EnumerateFiles(ALPHABET_PATH, "*.png");
            var loadSingle = (string filePath) =>
            {
                string maskPath = filePath.Replace(".png", ".mask.png");
                Image<Gray, byte>? mask = null;
                if (File.Exists(maskPath))
                {
                    mask = new Image<Bgr, Byte>(maskPath).Convert<Gray, byte>();
                }

                var gray = new Image<Bgr, Byte>(filePath).Convert<Hsv, byte>().Split()[2];

                return (gray, mask);
            };

            foreach (var filePath in files)
            {
                string strval = "";
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

        public List<ChatLineContent> ProcessChatScreen(Image<Bgr, byte> img)
        {
            List<ChatLineContent> resultLines = new List<ChatLineContent>();
            var gray = img.Convert<Hsv, byte>().Split()[2];

            Rectangle rect = new Rectangle();
            Mat outputMask = Mat.Zeros(gray.Height + 2, gray.Width + 2, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            CvInvoke.FloodFill(gray, outputMask, new Point(0, 0), new MCvScalar(0), out rect, new MCvScalar(20), new MCvScalar(20), Emgu.CV.CvEnum.Connectivity.EightConnected);

            Matrix<byte> row = new Matrix<byte>(img.Rows, 1);
            gray.Reduce<byte>(row, Emgu.CV.CvEnum.ReduceDimension.SingleCol, Emgu.CV.CvEnum.ReduceType.ReduceMax);

            var output = gray;

            //Matrix<byte> rowFiltered = new Matrix<byte>(img.Rows, 1);
            //CvInvoke.Threshold(row, rowFiltered, 90, 255, Emgu.CV.CvEnum.ThresholdType.BinaryInv);

            int lastPicBottom = 0;
            for (int i = 0; i < row.Rows; i++)
            {
                var height = i - lastPicBottom;
                if (row[i, 0] < 110)
                {
                    if (height > 15)
                    {
                        var line = gray.GetSubRect(new Rectangle(0, lastPicBottom, img.Width, height));
                        output.Draw(new LineSegment2D(new Point(0, i), new Point(img.Width, i)), new Gray(255), 1);

                        resultLines.Add(ProcessChatLine(line));
                    }
                    lastPicBottom = i;
                }
            }

            //CvInvoke.Imshow("input", output);
            //CvInvoke.WaitKey(50);

            return resultLines;
        }

        private Point? FindTemplate(Image<Gray, byte> img, Image<Gray, byte> template, Image<Gray, byte>? mask)
        {
            Mat result = new Mat();
            if (mask == null)
            {
                CvInvoke.MatchTemplate(img, template, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed);
            }
            else
            {
                CvInvoke.MatchTemplate(img, template, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed, mask);
            }

            double minVal = 0, maxVal = 0;
            Point minLoc = new Point(), maxLoc = new Point();
            CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc);

            if (minVal < TEMPLATE_MATCH_THREASHOLD)
            {
                return new Point(minLoc.X, minLoc.Y);
            }

            return null;
        }

        private List<Point> FindTemplates(Image<Gray, byte> img, Image<Gray, byte> template, Image<Gray, byte>? mask)
        {
            List<Point> matches = new List<Point>();

            Mat result = new Mat();
            if (mask == null)
            {
                CvInvoke.MatchTemplate(img, template, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed);
            }
            else
            {
                CvInvoke.MatchTemplate(img, template, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed, mask);
            }

            var minmaxMask = Mat.Ones(result.Rows, result.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            while (true)
            {
                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc, minmaxMask);

                if (minVal < TEMPLATE_MATCH_THREASHOLD)
                {
                    int x = minLoc.X;
                    int y = minLoc.Y;
                    int w = template.Width;
                    int h = template.Height;
                    Rectangle rect = new Rectangle(x, y, w, h);

                    CvInvoke.Rectangle(minmaxMask, rect, new MCvScalar(0), -1);

                    matches.Add(new Point(x, y));
                }
                else
                {
                    break;
                }
            }

            return matches;
        }

        private Image<Gray, byte> GetSubRect(Image<Gray, byte> img, int start, int end)
        {
            return img.GetSubRect(new Rectangle(start, 0, end - start, img.Height));
        }

        public ChatLineContent ProcessChatLine(Image<Gray, byte> img)
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
