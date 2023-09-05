# ClaviRuntime
### Introduction <br>
This repository offers a C# library for CLAVIRuntime, enabling users to leverage the CLAVI AI Model for image processing, perform inference with customizable parameters, and seamlessly integrate it into your own applications.
### Constructors
| Classes | Description |
| --- | --- |
| Anomaly | Examine specific data points and detect rare occurrences that are suspicious |
| FaceRecognition | Coming soon |
| ImageClassification | Categorize and label groups of pixels or vectors within an image | 
| InstanceSegmentation | Identify and separate individual objects within an image |
| ObjectDetection | Locates and categorizes entities within images | 
| SemanticSegmentation | Categorize each pixel in an image into a class or object |<br>

See .NET Documentation

## How to install and manage CLAVIRuntime NuGet package in Visual Studio (locally)<br>
1. Download [ClaviRuntime.1.0.3.nupkg](https://github.com/RVisionSystem/ClaviRuntime/blob/master/ClaviRuntime.1.0.3.nupkg).
2. Create a new folder named ```CLAVIRuntime``` in ```Program Files (x86)\Microsoft SDKs\NuGetPackages```, create a subfolder within it, and name the subfolder with downloaded package version ```1.0.3```, paste file ```ClaviRuntime.1.0.3.nupkg```within the folder.<br>
   The hierarchical package folder tree has the following structure:
   
   ```
   Program Files (x86)
   └─ Microsoft SDKs
      └─ NuGetPackages
         └─ CLAVIRuntime
           └─ 1.0.3
             └─ ClaviRuntime.1.0.3.nupkg
   ```
3. Open your project in Visual Studio, in **Solution Explorer**, and then select **Project** > **Manage NuGet Packages**.
4. Select the **Browse** the tab. To search for a specific package, select **Settings** icon in the right pane.
   
![Screenshot 2023-09-05 092439](https://github.com/RVisionSystem/ClaviRuntime/assets/66403375/4bb9537d-b0ac-4cd2-9386-d2b330727d5a)

5. In the Options window, expand the NuGet Package Manager node and select Package Sources.
6. Select **+**, edit the **Name** as ```CLAVI Package```, browse the path in **Source** to the package file from step 2 ```Program Files (x86)\Microsoft SDKs\NuGetPackages\CLAVIRuntime\1.0.3\ClaviRuntime.1.0.3.nupkg```, and then select **OK**.

![Screenshot 2023-09-05 094103](https://github.com/RVisionSystem/ClaviRuntime/assets/66403375/45387e5c-058d-4e32-9c99-0a6a04ac8c10)
  
7. At the right pane, select **Package source** dropdown, select **CLAVI Package**, CLAVIRuntime will appear in the left pane, select and install the package. 

![Screenshot 2023-09-05 095120](https://github.com/RVisionSystem/ClaviRuntime/assets/66403375/acd634a4-6b55-4a43-b7c4-52adc6c62078)

## Example
#### Importing the library
```csharp
   using ClaviRuntime;
```
#### Creating the inference method
```csharp
  string modelPath = "<Path to your model file>";
  string imagePath = "<Path to your image file>";

  var classify = new ImageClassification();
  classify.InitializeModel(modelPath);
  classify.Process(imagePath);
  List<ImageClassificationResults> results = classify.resultsList;
  if (results.Count != 0)
  {
      foreach (var r in results)
      {
          Console.WriteLine(r.Name);
          Console.WriteLine(r.Score);
      }
  }
```
Find more examples see. 




