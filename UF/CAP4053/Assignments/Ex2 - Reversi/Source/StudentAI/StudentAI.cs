using GameAI.GamePlaying.Core;

namespace GameAI.GamePlaying
{
    public class StudentAI : Behavior
    {
        private int[,] PostionScore;
        private int[,] PostionScoreExample;
        int positionalScoringFactor;
        int evaporaitonScoringFactor;
        int mobilityScoringFactor;
        int terminalScoringFactor;

        public StudentAI()
        {
            PostionScore = new int[,] {
              {100, 1, 10, 10, 10, 10, 1, 100},
              {1, 1, 2, 2, 2, 2, 2, 1},
              {10, 2, 2, 2, 2, 2, 2, 10},
              {10, 2, 2, 2, 2, 2, 2, 10},
              {10, 2, 2, 2, 2, 2, 2, 10},
              {10, 2, 2, 2, 2, 2, 2, 10},
              {1, 1, 2, 2, 2, 2, 2, 1},
              {100, 1, 10, 10, 10, 10, 1, 100}
          };
            PostionScoreExample = new int[,] {
              {100, 10, 10, 10, 10, 10, 10, 100},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {10, 1, 1, 1, 1, 1, 1, 10},
              {100, 10, 10, 10, 10, 10, 10, 100}
          };
            positionalScoringFactor = 1;
            evaporaitonScoringFactor = 10;
            mobilityScoringFactor = 10;
            terminalScoringFactor = 10000;
        }

        public ComputerMove Run(int _color, Board _board, int _lookAheadDepth)
        {
            // Initialize vars
            ComputerMove bestMove = null;
            Board newBoard = new Board();

            // Loop through possible moves
            for (int row = 0; row < 8; ++row)
            {
                for (int col = 0; col < 8; ++col)
                {
                    if (_board.IsValidMove(_color, row, col))
                    {
                        // Make move
                        newBoard.Copy(_board);
                        newBoard.MakeMove(_color, row, col);
                        ComputerMove newMove = new ComputerMove(row, col);

                        // Check if terminal
                        if (newBoard.IsTerminalState() || _lookAheadDepth < 1)
                        {
                            newMove.rank = EvaluateExample(newBoard);
                        }
                        else
                        {
                            newMove.rank = Run(GetNextPlayer(_color, newBoard), newBoard, _lookAheadDepth - 1).rank;
                        }

                        if (bestMove == null || IsMoveBetter(_color, newMove, bestMove))
                        {
                            bestMove = newMove;
                        }
                    }
                }
            }
            return bestMove;
        }

        private int EvaluateExample(Board board)
        {
            int score = 0;
            for (int row = 0; row < 8; ++row)
            {
                for (int col = 0; col < 8; ++col)
                {
                    score += board.GetTile(row, col) * PostionScoreExample[row, col];
                }
            }
            if (board.IsTerminalState())
            {
                if (score > 0) { score += terminalScoringFactor; }
                else if (score < 0) { score -= terminalScoringFactor; }
            }
            return score;
        }

        private int Evaluate(Board board, int player)
        {
            if (player == 1) { return EvaluateExample(board); }
            // https://samsoft.org.uk/reversi/strategy.htm
            // Positional scoring
            int score = 0;
            for (int row = 0; row < 8; ++row)
            {
                for (int col = 0; col < 8; ++col)
                {
                    if (board.EmptyCount < 60)
                    {
                        score += board.GetTile(row, col) * PostionScore[row, col];
                    }
                    else
                    {
                        score += board.GetTile(row, col);
                    }
                    
                }
            }
            score *= positionalScoringFactor;
            
            // Terminal scoring
            if (board.IsTerminalState())
            {
                if (score > 0) { score += terminalScoringFactor; }
                else if (score < 0) { score -= terminalScoringFactor; }
            }
            return score;
        }

        private int EvaporationScoring(Board board, int player)
        {
            if (board.EmptyCount > 40)
            {
                return player * evaporaitonScoringFactor * (board.WhiteCount - board.BlackCount);
            }
            else
            {
                return player * evaporaitonScoringFactor * (board.BlackCount - board.WhiteCount);
            }
        }

        private int MobilityScoring(Board board, int player)
        {
            int whiteMoves = 0;
            for (int row = 0; row < 8; ++row)
            {
                for (int col = 0; col < 8; ++col)
                {
                    if (board.IsValidMove(-1, row, col))
                    {
                        ++whiteMoves;
                    }
                }
            }
            return mobilityScoringFactor * whiteMoves;
        }

        private int GetNextPlayer(int player, Board board)
        {
            if (board.HasAnyValidMove(-player)) { return -player;  }
            else {  return player; }
        }

        private bool IsMoveBetter(int player, ComputerMove newMove, ComputerMove bestMove)
        {
            if (player == 1) { return newMove.rank > bestMove.rank; }
            else { return newMove.rank < bestMove.rank; }
        }
    }
}
