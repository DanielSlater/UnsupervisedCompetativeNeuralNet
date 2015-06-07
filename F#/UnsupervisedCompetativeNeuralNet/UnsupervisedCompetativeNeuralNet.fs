module UnsupervisedCompetativeNeuralNet

[<StructuredFormatDisplay("Particle {Parameters}")>]
type Parameters =
    val bias : float[]
    val connectionWeights : float[][]
    val input_nodes : int
    val hidden_nodes : int
    val mutable learning_rate : float
    val mutable bias_learning_rate : float
    new (input_nodes, hidden_nodes, learning_rate, bias_learning_rate) =
        let random = System.Random()
        {
        bias = [|for i in 0 .. hidden_nodes do yield 0.0|];
        connectionWeights = [|for i in 0 .. input_nodes do 
                                yield [|for j in 0 .. hidden_nodes do
                                            yield random.NextDouble()
                                      |]
                            |];
        input_nodes = input_nodes;
        hidden_nodes=hidden_nodes;
        learning_rate = learning_rate;
        bias_learning_rate = bias_learning_rate
        }

let max_index (array : float[]) =
    let mutable max_value = array.[0]
    let mutable max_index = 0
    for i in 1 .. array.Length do
        if array.[i] > max_value then
            max_value <- array.[i]
            max_index <- i
    max_index
        

let feed_forward (parameters : Parameters) (data : float[]) =
    let input_activations = data
    [| for h in 0 .. parameters.hidden_nodes do
        yield parameters.bias.[h] + (data |> Array.mapi (fun i value -> parameters.connectionWeights.[i].[h]*value) |> Array.sum)
    |]

let train parameters (data : float[]) =
    let hidden_nodes = feed_forward parameters data
    let winning_node = hidden_nodes |> max_index
    for i in 0 .. parameters.input_nodes do
        parameters.connectionWeights.[i].[winning_node] <- parameters.connectionWeights.[i].[winning_node] 
                                                         + parameters.learning_rate*(data.[i]-parameters.connectionWeights.[i].[winning_node])
    parameters.bias.[winning_node] <- parameters.bias.[winning_node]-parameters.bias_learning_rate
    winning_node

let train_many parameters (dataSet : seq<float[]>) =
    for data in dataSet do
        train parameters data |> ignore

let get_cluster parameters data =
    let hidden_nodes = feed_forward parameters data
    hidden_nodes |> max_index