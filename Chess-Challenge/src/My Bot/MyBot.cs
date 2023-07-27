using ChessChallenge.API;
using System;
using System.Linq;


public class MyBot : IChessBot
{
    private NeuralNetwork network;
    public Move Think(Board board, Timer timer)
    {
        if (network == null)
        {
            // Load the network from a file
            // network = NeuralNetwork.Load("network.json");
            Random random = new Random();

            int inputLayerSize = 389;
            int hiddenLayer1Size = 64;
            int hiddenLayer2Size = 64;
            int outputLayerSize = 3;

            // Initialize weights
            double[][][] weights = new double[3][][];

            weights[0] = GenerateRandomArray(hiddenLayer1Size, inputLayerSize, random); // 64
            weights[1] = GenerateRandomArray(hiddenLayer2Size, hiddenLayer1Size, random); // 64
            weights[2] = GenerateRandomArray(outputLayerSize, hiddenLayer2Size, random); // 3

            // Initialize biases
            double[][] biases = new double[3][];

            biases[0] = GenerateRandomArray(hiddenLayer1Size, random); // 64
            biases[1] = GenerateRandomArray(hiddenLayer2Size, random); // 64
            biases[2] = GenerateRandomArray(outputLayerSize, random); // 3

        // Initialize network
            network = new NeuralNetwork(weights, biases);
        }
        int xxxxxxxxxxx;
        String s = nameof(xxxxxxxxxxx);
        // print s
        Console.WriteLine(s);
        Move[] moves = board.GetLegalMoves();
        double[] scores = new double[moves.Length];
        // loop over all legal moves
        for (int i = 0; i < moves.Length; i++)
        {
            // Make the move on the copy
            board.MakeMove(moves[i]);
            // Get the board state after the move
            double[] boardStateCopy = new double[6 * 64 + 5];
            for(int j = 0; j < 6; j++) 
            {   
                ulong whiteBitboard = board.GetPieceBitboard((PieceType)j, true);
                ulong blackBitboard = board.GetPieceBitboard((PieceType)j, false);
                for(int k = 0; k < 64; k++) 
                {
                    int index = j * 64 + k;
                    boardStateCopy[index] += (double)((whiteBitboard >> k) & 1);
                    boardStateCopy[index] -= (double)((blackBitboard >> k) & 1);
                }
            }
            boardStateCopy[6 * 64] = board.IsWhiteToMove ? 1f : 0;
            boardStateCopy[6 * 64 + 1] = board.HasKingsideCastleRight(true) ? 1f : 0;
            boardStateCopy[6 * 64 + 2] = board.HasQueensideCastleRight(true) ? 1f : 0;
            // Now for blackjee
            boardStateCopy[6 * 64 + 3] = board.HasKingsideCastleRight(false) ? 1f : 0;
            boardStateCopy[6 * 64 + 4] = board.HasQueensideCastleRight(false) ? 1f : 0;

            // Feed the board state to the network
            double[] output = network.FeedForward(boardStateCopy);
            // Get the score for the move
            scores[i] = output[0] - output[1];

            // Undo the move
            board.UndoMove(moves[i]);
        }

        // play the move with the highest score using max
        return moves[scores.ToList().IndexOf(scores.Max())];
    }
    static double[][] GenerateRandomArray(int rows, int cols, Random random)
    {
        double[][] array = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            array[i] = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                // Generates a random double between 0.0 and 1.0
                array[i][j] = random.NextDouble();
            }
        }
        return array;
    }

    static double[] GenerateRandomArray(int size, Random random)
    {
        double[] array = new double[size];
        for (int i = 0; i < size; i++)
        {
            // Generates a random double between 0.0 and 1.0
            array[i] = random.NextDouble();
        }
        return array;
    }
}

public class NeuralNetwork
{
    private double[][][] weights;
    private double[][] biases;

    public NeuralNetwork(double[][][] weights, double[][] biases)
    {
        this.weights = weights;
        this.biases = biases;
    }

    private static double ReLU(double x)
    {
        return Math.Max(0, x);
    }

    private static double[] Softmax(double[] values)
    {
        double maxVal = values.Max();
        double scale = values.Sum(x => Math.Exp(x - maxVal));
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Exp(values[i] - maxVal) / scale;
        }
        return values;
    }

    public double[] FeedForward(double[] input)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            double[] layerOutput = new double[weights[i].Length];
            for(int j = 0; j < weights[i].Length; j++)
            {
                double sum = 0;
                for(int k = 0; k < weights[i][j].Length; k++)
                {
                    sum += input[k] * weights[i][j][k];
                }
                layerOutput[j] = ReLU(sum + biases[i][j]);
            }
            input = layerOutput;
        }

        // Apply Softmax function to the output layer
        input = Softmax(input);

        return input;
    }
}