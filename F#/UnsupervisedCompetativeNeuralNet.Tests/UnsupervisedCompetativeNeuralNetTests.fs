namespace UnsupervisedCompetativeNeuralNet.Tests
open UnsupervisedCompetativeNeuralNet
open NUnit.Framework
open System.Linq

[<TestFixture>]
type UnsupervisedCompetativeNeuralNetTests() = 
    let validate_clusters (data_set : float[][]) (expected_clusters : int[][]) (parameters : Parameters) =
        expected_clusters |> Array.

    let train_on_data_set (data_set : float[][]) (expected_clusters : int[][]) =
        let parameters = Parameters(data_set.[0].Length, expected_clusters.Length, 0.2, 0.01)
        for i in 0 .. 100 do
            train_many parameters data_set

            if 

    [<Test>]
    member this.ToyDataSet() = 
        let data_set = [| [|1.0; 0.9; 0.8|];
                          [|0.0; 0.1; 0.2|];
                          [|0.2; 0.1; 0.1|];
                          [|0.9; 0.9; 0.8|];
                        |]
        let expected_clusters = [| [|0; 3|];
                                   [|1; 2|]
                               |]
        Assert.IsTrue(train_on_data_set data_set expected_clusters)