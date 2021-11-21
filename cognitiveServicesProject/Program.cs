using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.Azure.CognitiveServices.Vision.Face;
using Microsoft.Azure.CognitiveServices.Vision.Face.Models;


// From your Face subscription in the Azure portal, get your subscription key and endpoint.
const string SUBSCRIPTION_KEY = "";
const string ENDPOINT = "";
// Used for all examples.
// URL for the images.
const string IMAGE_BASE_URL = "https://csdx.blob.core.windows.net/resources/Face/Images/";

namespace cognitiveServicesProject
{
    class Program
    {
        static string personGroupId = Guid.NewGuid().ToString();
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            // Authenticate.
            IFaceClient client = Authenticate(ENDPOINT, SUBSCRIPTION_KEY);

            // Detect - get features from faces.
            DetectFaceExtract(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();
            // Find Similar - find a similar face from a list of faces.
            FindSimilar(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();
            // Verify - compare two images if the same person or not.
            Verify(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();

            // Identify - recognize a face(s) in a person group (a person group is created in this example).
            IdentifyInPersonGroup(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();
            // LargePersonGroup - create, then get data.
            LargePersonGroup(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();
            // Group faces - automatically group similar faces.
            Group(client, IMAGE_BASE_URL, RECOGNITION_MODEL4).Wait();
            // FaceList - create a face list, then get data
            FaceListOperations(client, IMAGE_BASE_URL).Wait();
            // Large FaceList - create a large face list, then get data
            LargeFaceListOperations(client, IMAGE_BASE_URL).Wait();



            // Recognition model 4 was released in 2021 February.
            // It is recommended since its accuracy is improved
            // on faces wearing masks compared with model 3,
            // and its overall accuracy is improved compared
            // with models 1 and 2.
            const string RECOGNITION_MODEL4 = RecognitionModel.Recognition04;


        }

        /*
        *	AUTHENTICATE
        *	Uses subscription key and region to create a client.
        */
        public static IFaceClient Authenticate(string endpoint, string key)
        {
            return new FaceClient(new ApiKeyServiceClientCredentials(key)) { Endpoint = endpoint };
        }

        public static async Task IdentifyInPersonGroup(IFaceClient client, string url, string recognitionModel)
        {
            Console.WriteLine("========IDENTIFY FACES========");
            Console.WriteLine();

            // Create a dictionary for all your images, grouping similar ones under the same key.
            Dictionary<string, string[]> personDictionary =
                new Dictionary<string, string[]>
                    { { "Family1-Dad", new[] { "Family1-Dad1.jpg", "Family1-Dad2.jpg" } },
              { "Family1-Mom", new[] { "Family1-Mom1.jpg", "Family1-Mom2.jpg" } },
              { "Family1-Son", new[] { "Family1-Son1.jpg", "Family1-Son2.jpg" } },
              { "Family1-Daughter", new[] { "Family1-Daughter1.jpg", "Family1-Daughter2.jpg" } },
              { "Family2-Lady", new[] { "Family2-Lady1.jpg", "Family2-Lady2.jpg" } },
              { "Family2-Man", new[] { "Family2-Man1.jpg", "Family2-Man2.jpg" } }
                    };
            // A group photo that includes some of the persons you seek to identify from your dictionary.
            string sourceImageFileName = "identification1.jpg";

            // Create a person group. 
            Console.WriteLine($"Create a person group ({personGroupId}).");
            await client.PersonGroup.CreateAsync(personGroupId, personGroupId, recognitionModel: recognitionModel);
            // The similar faces will be grouped into a single person group person.
            foreach (var groupedFace in personDictionary.Keys)
            {
                // Limit TPS
                await Task.Delay(250);
                Person person = await client.PersonGroupPerson.CreateAsync(personGroupId: personGroupId, name: groupedFace);
                Console.WriteLine($"Create a person group person '{groupedFace}'.");

                // Add face to the person group person.
                foreach (var similarImage in personDictionary[groupedFace])
                {
                    Console.WriteLine($"Add face to the person group person({groupedFace}) from image `{similarImage}`");
                    PersistedFace face = await client.PersonGroupPerson.AddFaceFromUrlAsync(personGroupId, person.PersonId,
                        $"{url}{similarImage}", similarImage);
                }
            }
            // Start to train the person group.
            Console.WriteLine();
            Console.WriteLine($"Train person group {personGroupId}.");
            await client.PersonGroup.TrainAsync(personGroupId);

            // Wait until the training is completed.
            while (true)
            {
                await Task.Delay(1000);
                var trainingStatus = await client.PersonGroup.GetTrainingStatusAsync(personGroupId);
                Console.WriteLine($"Training status: {trainingStatus.Status}.");
                if (trainingStatus.Status == TrainingStatusType.Succeeded) { break; }
            }
            Console.WriteLine();
            List<Guid> sourceFaceIds = new List<Guid>();
            // Detect faces from source image url.
            List<DetectedFace> detectedFaces = await DetectFaceRecognize(client, $"{url}{sourceImageFileName}", recognitionModel);

            // Add detected faceId to sourceFaceIds.
            foreach (var detectedFace in detectedFaces) { sourceFaceIds.Add(detectedFace.FaceId.Value); }
            // Identify the faces in a person group. 
            var identifyResults = await client.Face.IdentifyAsync(sourceFaceIds, personGroupId);

            foreach (var identifyResult in identifyResults)
            {
                Person person = await client.PersonGroupPerson.GetAsync(personGroupId, identifyResult.Candidates[0].PersonId);
                Console.WriteLine($"Person '{person.Name}' is identified for face in: {sourceImageFileName} - {identifyResult.FaceId}," +
                    $" confidence: {identifyResult.Candidates[0].Confidence}.");
            }
            Console.WriteLine();
        }

        /*
        * DELETE PERSON GROUP
        * After this entire example is executed, delete the person group in your Azure account,
        * otherwise you cannot recreate one with the same name (if running example repeatedly).
        */
        public static async Task DeletePersonGroup(IFaceClient client, String personGroupId)
        {
            await client.PersonGroup.DeleteAsync(personGroupId);
            Console.WriteLine($"Deleted the person group {personGroupId}.");
        }
    }
}
