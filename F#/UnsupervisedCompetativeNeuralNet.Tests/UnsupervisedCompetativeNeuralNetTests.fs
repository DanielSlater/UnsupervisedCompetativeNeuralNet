namespace UnsupervisedCompetativeNeuralNet.Tests
open UnsupervisedCompetativeNeuralNet
open NUnit.Framework
open System.Linq

[<TestFixture>]
type UnsupervisedCompetativeNeuralNetTests() = 
    let validate_clusters (data_set : float[][]) (expected_clusters : int[][]) (parameters : Parameters) =
        expected_clusters |> Seq.forall (fun x -> x |> Seq.groupBy (fun x -> get_cluster parameters data_set.[x]) 
                                                    |> Seq.length = 1)

    let rec train_iterations data_set expected_clusters parameters iterations =
        train_many parameters data_set
        if validate_clusters data_set expected_clusters parameters then
            true
        elif iterations <= 0 then
            false
        else
            train_iterations data_set expected_clusters parameters (iterations - 1)

    let train_on_data_set (data_set : float[][]) (expected_clusters : int[][]) =
        let parameters = Parameters(data_set.[0].Length, expected_clusters.Length, 0.2, 0.01)
        if train_iterations data_set expected_clusters parameters 100 then
            true
        else
            for item in data_set do
                let cluster = get_cluster parameters item
                printf "%A %i" item cluster
            false

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

    [<Test>]
    member this.OnlyASingleValueMatters() =
            let data_set = [|
                [|1.0; 0.9; 1.0|];
                [|1.0; 1.0; 0.0|];
                [|1.0; 0.9; 0.0|];
                [|0.9; 1.0; 1.0|];
                    |]
            let expected_clusters = [|
                [|0; 3|];
                [|1; 2|];
                    |]

            Assert.IsTrue(train_on_data_set data_set expected_clusters)

        [<Test>]
        member this.ThreeClasses() =
            let data_set = [|
                [|0.2; 0.9|];
                [|0.2; 0.8|];
                [|0.9; 0.9|];
                [|0.8; 0.9|];
                [|0.5; 0.1|];
                [|0.3; 0.0|];
                        |]
            let expected_clusters = [|
                [|0; 1|];
                [|2; 3|];
                [|4; 5|];
                    |]

            Assert.IsTrue(train_on_data_set data_set expected_clusters)