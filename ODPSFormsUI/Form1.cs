using LiveChartsCore.Defaults;
using System.Collections.ObjectModel;
using ODPSCore;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;

namespace ODPSFormsUI
{
    public partial class Form1 : Form
    {
        private ODPS dpsMeasure;
        ObservableCollection<ObservableValue> m_data = new ObservableCollection<ObservableValue>();
        ObservableCollection<ISeries> m_series;

        public Form1()
        {
            m_series = new ObservableCollection<ISeries>
            {
                new LineSeries<ObservableValue>
                {
                    Values = m_data,
                }
            };
            InitializeComponent();
            this.dpsChart.Series = m_series;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            dpsMeasure = new ODPS();
            dpsMeasure.DpsChanged += DpsMeasure_DpsChanged;
        }

        private void DpsMeasure_DpsChanged(object? sender, DpsInfo e)
        {
            m_data.Add(new ObservableValue(e.total / e.duration.TotalSeconds));
            if (m_data.Count > 30)
            {
                m_data.RemoveAt(0);
            }
        }
    }
}