+++
title = "cockfish"
date = "2026-01-19T21:07:23Z"
#dateFormat = "2006-01-02" # This value can be configured for per-post date formatting
author = ""
authorTwitter = "" #do not include @
cover = ""
tags = ["ml", "chess", "neural networks"]
keywords = ["", ""]
description = "training a chess engine from scratch"
showFullContent = false
readingTime = true
hideComments = false
+++

chess engines are very strong nowadays, and work through mysterious magic. courtsey of chatgpt:

> Stockfish works by searching billions of chess positions per second using alpha-beta–pruned minimax guided by finely tuned evaluation heuristics (and NNUE), and it’s so effective because this combination turns raw computing power into extremely accurate position assessment and move selection.

basically, if we have some way of quantifying how good a position is in a score (an evaluation function), and we have an efficient way to search move sequences (alpha-beta pruned minimax), then we can just search a bunch of moves and pick the ones that lead to the best outcome.

stockfish does the evaluation function with a big set of handcrafted weights, specifically encoding known advantages in chess like:
- king safety
- space advantages
- queens and rooks supporting each other
- central control
- yada yada yada...

my question is, can a neural network learn this stuff without being told it beforehand? i.e. if i'm some chump who doesn't know a thing about chess beyond the rules, can i make a reasonable engine?

# setup

## minimax

the minimax algorithm is a classic leetcode recursive depth first search problem, with the slight added complexity in the sense that when we pick a move to maximise our score, we have to assume the opponent in their turn is going to pick the move that maximises *theirs*, i.e. by minimising ours.

```python
def minimax(position, depth, maximizing_player):
    # base case: reached max depth or game over
    if depth == 0 or is_terminal(position):
        return evaluate(position)
    
    if maximizing_player:
        max_eval = -infinity
        for move in get_legal_moves(position):
            new_position = make_move(position, move)
            eval = minimax(new_position, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = infinity
        for move in get_legal_moves(position):
            new_position = make_move(position, move)
            eval = minimax(new_position, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval
```

the problem with this is that recursive functions, especially in chess where the number of legal moves is large, are slow.

we get around this with **caching**. chess boards can have a unique string representation, making global caching of positions that have already been seen, along with our estimated evaluation for it, already cached. if we run into a position where we know from prior search the opponent will have the advantage with optimal play, we can ignore searching it again.

## data
[lichess](lichess.org) kindly makes their games database available for download, so i downloaded 18gb of games + stockfish evaluations for december 2025 in a zstd-compressed JSONL file. this format allows me to stream it through memory chunk by chunk.

```python
class ZstdStreamer:
    """
    A class for streaming the Lichesss DB Eval dataset from a Zstandard-compressed JSONL file.
    Yields chunks of parsed JSON objects
    """

    def __init__(self, path: str):
        self.path = path
        self.dctx = zstandard.ZstdDecompressor()

    def __iter__(self):
        """Yields parsed position dictionaries from the JSONL file."""
        with open(self.path, 'rb') as file:
            with self.dctx.stream_reader(file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
```

this can be plugged into a pytorch dataloader that uses the python-chess library to generate a chess board from the position FEN string, and yields batches of data to the neural net for training.

```python
class LichessEvalDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming Lichess evaluation data.
    
    Streams positions from compressed JSONL file, converts to tensors on-the-fly.
    Returns raw centipawn evaluations - apply normalization in training code.
    No preprocessing or decompression to disk required.
    
    Uses a shuffle buffer to randomize the order of positions, breaking up
    correlations between consecutive positions from the same game.
    
    Note: This dataset yields positions for a SINGLE epoch (one pass through the file).
    To train for multiple epochs, create a new iterator or call the training loop multiple times.
    """
    
    def __init__(
        self, 
        zstd_path: str,
        max_positions: Optional[int] = None,
        shuffle_buffer_size: int = 10000
    ):
        """
        Args:
            zstd_path: Path to lichess_db_eval.jsonl.zst file
            max_positions: Optional limit on number of positions to yield per epoch
            shuffle_buffer_size: Size of the shuffle buffer (larger = better shuffling, more memory)
                                 Set to 0 to disable shuffling
        """
        self.zstd_path = zstd_path
        self.max_positions = max_positions
        self.shuffle_buffer_size = shuffle_buffer_size
        self.positions_in_last_epoch = 0  # Track how many positions were in the last epoch
    
    def __iter__(self):
        """Yields (board_tensor, centipawn_eval) pairs with optional shuffling."""
        streamer = ZstdStreamer(self.zstd_path)
        count = 0
        
        # Shuffle buffer for randomizing order
        buffer = []
        
        for record in streamer:
            if self.max_positions and count >= self.max_positions:
                break
            
            try:
                # Parse FEN and convert to tensor
                board = Board.from_fen(record['fen'])
                board_tensor = board.to_tensor()
                
                # Extract centipawn evaluation from first PV (raw value)
                cp = record['evals'][0]['pvs'][0]['cp']
                
                item = (board_tensor, torch.tensor(float(cp), dtype=torch.float32))
                
                if self.shuffle_buffer_size > 0:
                    # Add to buffer
                    buffer.append(item)
                    
                    # When buffer is full, yield a random item
                    if len(buffer) >= self.shuffle_buffer_size:
                        idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(idx)
                        count += 1
                else:
                    # No shuffling, yield directly
                    yield item
                    count += 1
                
            except (KeyError, IndexError, ValueError) as e:
                # Skip malformed records
                continue
        
        # Yield remaining items in buffer (shuffled)
        random.shuffle(buffer)
        for item in buffer:
            if self.max_positions and count >= self.max_positions:
                break
            yield item
            count += 1
        
        # Track positions yielded this epoch
        self.positions_in_last_epoch = count
```

## evaluation

i don't want to train a chess engine with any fancy chess knowledge hoohah. this is boring and people already spent a lot of time figuring it out in the 1800s. i want cockfish to learn from first principles straight from the womb.

this means some tensor representation of the chess board should be the only input, which i am representing as a 13x8x8 bitmap, an 8x8 chess board layer for each of the 12 pieces (black and white pieces being on different layers), and a single boolean 8x8 layer to show who's move it is (this is probably better to plug in as a second single input but oh well).

```python
from typing import Self, Optional
from dataclasses import dataclass

import torch
import numpy as np
from chess import Board as ChessBoard, WHITE

@dataclass
class Board:
    """
    The representation of our chess board.
    """
    fen: str
    board_obj: ChessBoard

    @classmethod
    def new_game(cls) -> Self:
        """
        Create a new board in the starting position.
        """
        return cls.from_fen(ChessBoard().fen())

    @classmethod
    def from_fen(cls, fen: str) -> Self:
        """
        Load a board from a FEN string.
        """
        chess_board = ChessBoard(fen)
        return cls(fen=fen, board_obj=chess_board)

    @property
    def legal_moves(self):
        """
        Get the legal moves for the current position.
        """
        return list(self.board_obj.legal_moves)
    
    def to_tensor(self) -> torch.Tensor:
        """
        Convert board to raw tensor representation.
        Returns a tensor of shape (13, 8, 8):
        - 12 planes for pieces (6 piece types x 2 colors)
        - 1 plane for side to move (all 1s if white, all 0s if black)
        """
        # Initialize 13 planes of 8x8
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        
        # Piece type mapping: pawn=1, knight=2, bishop=3, rook=4, queen=5, king=6
        # Planes 0-5: white pieces (P, N, B, R, Q, K)
        # Planes 6-11: black pieces (p, n, b, r, q, k)
        
        for square in range(64):
            piece = self.board_obj.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                
                # Determine plane index
                if piece.color == WHITE:
                    plane_idx = piece.piece_type - 1  # 0-5
                else:
                    plane_idx = piece.piece_type + 5  # 6-11
                
                planes[plane_idx, rank, file] = 1.0
        
        # Plane 12: side to move
        planes[12, :, :] = 1.0 if self.board_obj.turn == WHITE else 0.0
        
        return torch.from_numpy(planes)
```

## training

given the spacial setup of a chess board, a convolutional approach makes sense here, but we have to be careful. convolutions are great because they give us **local spatial awareness** — each neuron can see what pieces are around it and reason about tactical motifs like forks, pins, and skewers. 

but there's a catch: convolutions have a limited receptive field. without seeing the whole board at once, a network might miss strategic concepts like "i'm down a queen but my opponent's king is cornered" or "i have a passed pawn three squares from promotion". so we layer in **global pooling** — averaging features across the entire board and broadcasting them back to every square. this gives each position access to board-wide context without needing the receptive field to physically span 64 squares.

the backbone is **residual blocks** because they let us go deeper without gradient flow collapsing. each residual block takes in 64-dim spatial features and outputs 64-dim spatial features with a skip connection, so gradients can flow straight through. stack a few of these and the network can learn increasingly complex patterns.

after the convolutions squeeze out features, we flatten everything and throw it through a couple fully connected layers to predict a single number: the evaluation. we use tanh to squash it to [-1, 1], matching the normalized stockfish centipawn evaluations in our dataset.

```python
class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Allows gradients to flow directly through skip connection,
    enabling training of deeper networks.
    """
    
    def __init__(self, num_filters, activation='relu'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        # Store input for skip connection
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out = out + identity
        out = self.activation(out)
        
        return out


class CockfishNet(nn.Module):
    """
    Cockfish's Brain
    
    Takes raw board representation (13, 8, 8) and outputs a single evaluation score.
    Architecture: Conv layers => Residual blocks => Global context => FC layers => Single output
    """
    
    def __init__(
        self, 
        num_filters=64, 
        num_residual_blocks=4,
        use_global_pooling=True,
        activation='relu'
    ):
        """
        Args:
            num_filters: Number of filters in convolutional layers
            num_residual_blocks: Number of residual blocks (more = deeper, sees farther)
            use_global_pooling: Whether to add global board context to features
            activation: Activation function ('relu', 'gelu', 'silu', 'leaky_relu')
        """
        super().__init__()
        
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.use_global_pooling = use_global_pooling
        self.activation_name = activation
        
        # Get activation function
        self.activation = self._get_activation(activation)
        
        # Initial conv layer: 13 input channels (12 pieces + side to move) → num_filters
        self.conv_initial = nn.Conv2d(13, num_filters, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_filters)
        
        # Stack of residual blocks for deep feature extraction
        # Each block maintains spatial dimensions but learns increasingly complex patterns
        # More blocks = larger receptive field = can see farther across board
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters, activation) 
            for _ in range(num_residual_blocks)
        ])
        
        # Calculate size of flattened features
        if use_global_pooling:
            # Local features (spatial) + global features (broadcasted)
            # We concatenate them, so double the channels
            flatten_size = num_filters * 2 * 8 * 8
        else:
            # Just local spatial features
            flatten_size = num_filters * 8 * 8
        
        # Fully connected layers to combine all features into single evaluation
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Lower dropout - 0.3 was too aggressive for this task
        self.dropout = nn.Dropout(0.1)
    
    def _get_activation(self, name):
        """Get activation function by name."""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 13, 8, 8)
            
        Returns:
            Tensor of shape (batch_size,) - position evaluations
        """
        # Initial convolution to get base features
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = self.activation(x)
        
        # Pass through residual blocks
        # Each block refines features and expands receptive field
        for block in self.residual_blocks:
            x = block(x)
        
        # x is now (batch_size, num_filters, 8, 8)
        # Contains rich local features at each square
        
        if self.use_global_pooling:
            # Add global board context to help with long-range dependencies
            # Global pooling gives each square access to board-wide statistics
            # This helps capture things like "material balance" or "king safety"
            # without needing the receptive field to span entire board
            
            # Average all spatial positions to get global features
            global_features = x.mean(dim=[2, 3])  # (batch_size, num_filters)
            
            # Broadcast global features back to spatial dimensions
            # Now every square knows about the overall board state
            global_features = global_features.unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_filters, 1, 1)
            global_features = global_features.expand(-1, -1, 8, 8)  # (batch_size, num_filters, 8, 8)
            
            # Concatenate local and global features
            # Each position now has both local tactical info AND global strategic info
            x = torch.cat([x, global_features], dim=1)  # (batch_size, num_filters*2, 8, 8)
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output: single evaluation score, bounded to [-1, 1] to match tanh-normalized targets
        x = self.fc3(x)
        x = torch.tanh(x)
        
        return x.squeeze(-1)  # Shape: (batch_size,)
```

## results

chess is a closed game, and if we overfit to every possible game state, hey-ho, we have a great chess engine, so i've completely ignored validation loss here. 

if large language models have taught us anything, its that overfitting isn't an issue if you've overfit to every possible scenario.

i pumped it through a single epoch of the data, and after a couple of days of training on my macbook pro we get some nice training loss convergence.

![training_loss](/images/cockfish_loss.png)

to evaluate our engine, i set up a flask web app with a chess ui that lets me play the bot, and an auto play system to play it against various levels of stockfish and evaluate its ELO score (courtesy of chatgpt, as i'm too lazy to do this).

```
──────────────────────────────────────────────────────────────────────
OVERALL SUMMARY
Total Games: 110
Overall Score: 47.7% (+28 =49 -33)
Pooled Elo Estimate: 1633 ± 65
======================================================================
```

an ELO of 1633 puts cockfish somewhere around a strong club player level, it's beating most casual players and will give experienced amateurs a run for their money. to put it in perspective:

- 1200 ELO: a casual club player
- 1600 ELO: a strong club player
- 2000 ELO: master level
- 2700+ ELO: grandmaster

the interesting bit is that it got there with *literally no chess knowledge*. no piece-square tables, no king safety heuristics, no pawn structure evaluation - just raw board tensors and a network that had to figure out on its own that queens are worth roughly 9 pawns and that keeping your king safe is a pretty good idea.

i've played it many times now and can confirm it is annoying to play against. it doesn't play like stockfish (which is hyper-optimized and sometimes brutally computer-like), but rather like a really solid human player who understands positional play, doesn't hang pieces, and has some tactical awareness.

of course, it's not perfect. a grandmaster would tear it apart. but for something trained in a few days on a laptop with zero domain knowledge, it's pretty cool.


