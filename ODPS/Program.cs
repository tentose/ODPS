using Emgu.CV;
using Emgu.CV.Structure;
using ODPS;
using System.Diagnostics;
using System.Drawing;

// See https://aka.ms/new-console-template for more information

public partial class MainClass
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
    Dictionary<MarkerType, Image<Gray, byte>> markerImages = new Dictionary<MarkerType, Image<Gray, byte>>();

    enum MarkerSearchStrategy
    {
        RedChannel,
        ToGray,
    };

    private Point? LookForMarker(Image<Bgr, byte> img, MarkerType markerType, MarkerType mask, MarkerSearchStrategy searchStrategy, double confidenceThreshold = 0.3)
    {
        Point? markerPosition = null;
        var markerImg = markerImages[markerType];

        Image<Gray, byte> output = img.Convert<Gray, byte>();
        Gray boxColor = new Gray(255);

        Image<Gray, byte> searchImg = null;
        if (searchStrategy == MarkerSearchStrategy.ToGray)
        {
            searchImg = img.Convert<Gray, byte>();
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
            CvInvoke.MatchTemplate(searchImg, markerImg, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed);
        }
        else
        {
            CvInvoke.MatchTemplate(searchImg, markerImg, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed, markerImages[mask]);
        }

        result.MinMax(out double[] minVal, out double[] maxVal, out Point[] minLoc, out Point[] maxLoc);

        int x = minLoc[0].X;
        int y = minLoc[0].Y;
        int w = markerImg.Width;
        int h = markerImg.Height;
        Rectangle exclaimRect = new Rectangle(x, y, w, h);

        if (minVal[0] < confidenceThreshold)
        {
            markerPosition = minLoc[0];
        }

        return markerPosition;
    }

    string alphabetFileParentPath = @".\alphabet\";
    List<(string strval, Image<Gray, byte> img, Image<Gray, byte>? mask)> alphabet = new List<(string, Image<Gray, byte>, Image<Gray, byte>?)>();
    private void LoadAlphabet()
    {
        var files = Directory.EnumerateFiles(alphabetFileParentPath, "*.png");
        foreach (var filePath in files)
        {
            string strval = "";
            string fileName = Path.GetFileNameWithoutExtension(filePath);

            if (fileName.EndsWith(".mask"))
            {
                continue;
            }

            if (fileName.StartsWith("lower-"))
            {
                strval = fileName.Substring("lower-".Length);
            }
            else if (fileName == "colon")
            {
                strval = ":";
            }
            else
            {
                strval = fileName;
            }

            // is there a mask?
            string maskPath = filePath.Replace(".png", ".mask.png");
            Image<Gray, byte>? mask = null;
            if (File.Exists(maskPath))
            {
                mask = new Image<Bgr, Byte>(maskPath).Convert<Gray, byte>();
            }

            var original = new Image<Bgr, Byte>(filePath);
            var gray = original.Convert<Hsv, byte>().Split()[2];
            alphabet.Add((strval, gray, mask));
        }

        alphabet.Sort((a, b) =>
        {
            if (a.strval.Length != b.strval.Length)
            {
                return b.strval.Length.CompareTo(a.strval.Length);
            }
            else
            {
                return a.strval.CompareTo(b.strval);
            }
        });
    }

    public void SplitImageIntoLines(string fileName)
    {
        var outputFolderName = fileName.Replace(".png", "", StringComparison.InvariantCultureIgnoreCase);
        var colorOutputFolderName = outputFolderName + @"\line\";
        var grayOutputFolderName = outputFolderName + @"\gray\";
        Directory.CreateDirectory(colorOutputFolderName);
        Directory.CreateDirectory(grayOutputFolderName);

        var original = new Image<Bgr, Byte>(fileName);

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
        var img = original.GetSubRect(new Rectangle(left, top, right - left, bottom - top));
        //var gray = img.Convert<Gray, byte>();
        var gray = img.Convert<Hsv, byte>().Split()[2];
        var gray2 = gray;

        Rectangle rect = new Rectangle();
        Mat outputMask = new Mat(gray.Height + 2, gray.Width + 2, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
        CvInvoke.FloodFill(gray, outputMask, new Point(0, 0), new MCvScalar(0), out rect, new MCvScalar(20), new MCvScalar(20), Emgu.CV.CvEnum.Connectivity.EightConnected);

        CvInvoke.Imshow("input", gray);

        Stopwatch stopWatch = new Stopwatch();
        stopWatch.Start();

        //var lap = gray.Laplace(1).ConvertScale<byte>(1, 0);
        var output = gray;

        Matrix<byte> row = new Matrix<byte>(img.Rows, 1);
        gray.Reduce<byte>(row, Emgu.CV.CvEnum.ReduceDimension.SingleCol, Emgu.CV.CvEnum.ReduceType.ReduceMax);

        Matrix<byte> rowFiltered = new Matrix<byte>(img.Rows, 1);
        CvInvoke.Threshold(row, rowFiltered, 100, 255, Emgu.CV.CvEnum.ThresholdType.BinaryInv);

        int lineIndex = 0;
        int lastPicBottom = 0;
        for (int i = 0; i < rowFiltered.Rows; i++)
        {
            //Console.WriteLine($"{i}\t/{rowFiltered.Rows}\t: {row[i, 0]}\t = {rowFiltered[i, 0]}");

            var height = i - lastPicBottom;
            if (rowFiltered[i, 0] == 255)
            {
                if (height > 16)
                {
                    output.Draw(new LineSegment2D(new Point(0, i), new Point(img.Width, i)), new Gray(255), 1);

                    var line = gray2.GetSubRect(new Rectangle(0, lastPicBottom, img.Width, height));

                    var colorName = colorOutputFolderName + $"eng.gw2.exp{lineIndex}.png";
                    //line.Save(colorName);
                    //File.Move(colorName, colorName.Replace(".png", ""));

                    var lineGray = line.Convert<Gray, byte>();
                    var grayName = grayOutputFolderName + $"eng.gw2.exp{lineIndex}.png";
                    //lineGray.Save(grayName);
                    TryParseLine(lineGray);
                    //File.Move(grayName, grayName.Replace(".png", ""));

                    lineIndex++;
                }
                lastPicBottom = i;
            }
        }

        Console.WriteLine(stopWatch.ElapsedMilliseconds);
        stopWatch.Stop();

        CvInvoke.Imshow("output", output);
    }

    int cnt = 0;
    public void TryParseLine(Image<Gray, byte> img)
    {
        cnt++;
        Console.WriteLine("----------------");
        List<(int x, string s)> matches = new List<(int x, string s)>();
        foreach (var pair in alphabet)
        {
            if (pair.img.Height > img.Height)
            {
                Console.WriteLine("short line, skipping");
                break;
            }

            Mat result = new Mat();
            if (pair.mask == null)
            {
                CvInvoke.MatchTemplate(img, pair.img, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed);
            }
            else
            {
                CvInvoke.MatchTemplate(img, pair.img, result, Emgu.CV.CvEnum.TemplateMatchingType.SqdiffNormed, pair.mask);
            }



            var mask = Mat.Ones(result.Rows, result.Cols, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
            string location = "";
            while (true)
            {
                //result.MinMax(out double[] minVal, out double[] maxVal, out Point[] minLoc, out Point[] maxLoc);
                double minVal = 0, maxVal = 0;
                Point minLoc = new Point(), maxLoc = new Point();
                CvInvoke.MinMaxLoc(result, ref minVal, ref maxVal, ref minLoc, ref maxLoc, mask);

                if (minVal < 0.05)
                {
                    int x = minLoc.X;
                    int y = minLoc.Y;
                    int w = pair.img.Width;
                    int h = pair.img.Height;
                    Rectangle rect = new Rectangle(x, y, w, h);

                    location += $" {minVal} @ ({x}, {y}),";

                    CvInvoke.Rectangle(mask, rect, new MCvScalar(0), -1);

                    img.Draw(rect, new Gray(255), -1);

                    matches.Add((x, pair.strval));

                    if (cnt == 9)
                    {
                        CvInvoke.Imshow("line", img);
                    }
                    //break;
                }
                else
                {
                    break;
                }
                
            }

            //Console.WriteLine($"{pair.strval}: " + location);
        }

        matches.Sort((a, b) => a.x.CompareTo(b.x));
        foreach (var v in matches)
        {
            Console.Write(v.s + " ");
        }
        Console.WriteLine();
    }

    public void Exec()
    {
        // Load markers
        foreach (var kv in markerFilePaths)
        {
            if (!markerImages.ContainsKey(kv.Key))
            {
                var tmp = new Image<Bgr, Byte>(GetPathForMarker(kv.Key));
                var markerGray = tmp.Convert<Gray, byte>();
                markerImages[kv.Key] = markerGray;
            }
        }

        LoadAlphabet();

        
        var original = new Image<Bgr, Byte>(@".\sample\Uneven2.png");
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
        var img = original.GetSubRect(new Rectangle(left, top, right - left, bottom - top));
        //var gray = img.Convert<Gray, byte>();
        var gray = img.Convert<Hsv, byte>().Split()[2];

        ChatLineProcessor processor = new ChatLineProcessor();

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();
        for (int i = 0; i < 0; i++)
        {
            //var res = processor.ProcessChatScreen(img);
        }
        Console.WriteLine(stopwatch.ElapsedMilliseconds);
        stopwatch.Stop();

        //var result = processor.ProcessChatScreen(img);
        //foreach (var line in result)
        //{
        //    Console.WriteLine($"{line.Type}: {line.Value}");
        //}


        //SplitImageIntoLines(@".\sample\Uneven1.png");

        //CvInvoke.WaitKey(0);
        Console.ReadLine();
    }

    public static void Main()
    {
        //MainClass cls = new MainClass();
        //cls.Exec();

        TestCv cls = new TestCv();
        cls.Exec();

        //ODPS.ODPS dps = new ODPS.ODPS();

        //Console.ReadLine();
    }

}