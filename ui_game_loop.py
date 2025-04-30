if __name__ == "__main__":
    game = Game()
    depth = 3
    human_player = 1  # 1 = white, 2 = black

    print("CHECKERS - Human vs AI")
    print("Move format: Enter numbers like '0' for the first move in the list\n")

    while not game.is_over():
        writeBoard(game)
        
        if game.whose_turn() == human_player:
            # Human turn
            legal_moves = getAllFullMoves(game)
            
            if not legal_moves:
                print("No legal moves available!")
                break
                
            print("\nYour legal moves:")
            for i, move_obj in enumerate(legal_moves):
                print(f"{i}: Positions {move_obj.move_sequence}")

            while True:
                try:
                    choice = int(input("Enter move number: "))
                    if 0 <= choice < len(legal_moves):
                        legal_moves[choice].make_move(game)
                        break
                    else:
                        print(f"Invalid number. Enter 0-{len(legal_moves)-1}")
                except ValueError:
                    print("Numbers only please!")
        
        else:
            print("\nAI thinking...")
            start_time = timer()
            ai_move = minimaxStart(game, depth, maximizingPlayer=(human_player==2))
            ai_move.make_move(game)
            print(f"AI moved positions: {ai_move.move_sequence} (in {timer()-start_time:.1f}s)\n")

    writeBoard(game)
    winner = game.get_winner()
    print("\nGame Over!")
    print("Winner:", "Human" if winner==human_player else "AI" if winner else "Draw")