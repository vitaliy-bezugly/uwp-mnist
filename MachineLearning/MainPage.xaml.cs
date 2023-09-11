using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using Windows.Media;
using Windows.Storage.Streams;
using Windows.Storage;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;
using System.Linq;
using Microsoft.AI.MachineLearning;
using Windows.Foundation;
using System.Reflection;
using MachineLearning.Helpers;

namespace MachineLearning
{
    public sealed partial class MainPage : Page
    {
        private mnistModel mnistModelGen;
        private StringLengthModel stringLengthModelGen;
        private mnistInput mnistInput = new mnistInput();
        private mnistOutput mnistOutput;
        //private LearningModelSession    session;
        private MnistHelper helper = new MnistHelper();
        RenderTargetBitmap renderBitmap = new RenderTargetBitmap();

        public MainPage()
        {
            this.InitializeComponent();

            // Set supported inking device types.
            inkCanvas.InkPresenter.InputDeviceTypes = Windows.UI.Core.CoreInputDeviceTypes.Mouse | Windows.UI.Core.CoreInputDeviceTypes.Pen | Windows.UI.Core.CoreInputDeviceTypes.Touch;
            inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(
                new Windows.UI.Input.Inking.InkDrawingAttributes()
                {
                    Color = Windows.UI.Colors.White,
                    Size = new Size(22, 22),
                    IgnorePressure = true,
                    IgnoreTilt = true,
                }
            );

            LoadMnistModelAsync();
            LoadStringLengthModelAsync();
        }

        private async Task LoadMnistModelAsync()
        {
            //Load a machine learning model
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
            mnistModelGen = await mnistModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);
        }

        private async Task LoadStringLengthModelAsync()
        {
            //Load a machine learning model
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/string_length.onnx"));
            stringLengthModelGen = await StringLengthModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);
        }

        private async void recognizeButton_Click(object sender, RoutedEventArgs e)
        {
            //Bind model input with contents from InkCanvas
            VideoFrame vf = await helper.GetHandWrittenImage(inkGrid);
            mnistInput.Input3 = ImageFeatureValue.CreateFromVideoFrame(vf);

            //Evaluate the model
            mnistOutput = await mnistModelGen.EvaluateAsync(mnistInput);

            //Convert output to datatype
            IReadOnlyList<float> vectorImage = mnistOutput.Plus214_Output_0.GetAsVectorView();
            IList<float> imageList = vectorImage.ToList();

            //LINQ query to check for highest probability digit
            var maxIndex = imageList.IndexOf(imageList.Max());

            //Display the results
            numberLabel.Text = maxIndex.ToString();
        }

        private void clearButton_Click(object sender, RoutedEventArgs e)
        {
            inkCanvas.InkPresenter.StrokeContainer.Clear();
            numberLabel.Text = "";
        }

        private async void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            var stringHelper = new StringLengthHelper();

            // Prepare your input string
            var stringToCheck = InputTextBox.Text;
            var tensorLength = await stringHelper.GetStringTensor(stringToCheck);

            // Run the model
            var input = new StringLengthInput() { Length = tensorLength };
            var output = await stringLengthModelGen.EvaluateAsync(input);

            // Validate the output
            bool isValid = await stringHelper.IsStringLengthValid(output.Result);

            // Display the results
            OutputTextBlock.Text = isValid ? "Valid" : "Invalid";
        }
    }
}
