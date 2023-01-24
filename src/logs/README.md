# Description of Log

Please set **'-l'** to **True**, if you want to save information of the game.

The extracted logic state, all atoms and its probability are automatically saved in last csv file at log folder.


# Logic Representation

getout: [agent, key, door, enemy, X, Y]  
bigfish: [agent, fish, size, X, Y]  
bigfishcolor: [agent, fish, green, red, X, Y]  
heist: [agent, key, door, blue, red ,got_key, X, Y] 
heistdoors:   [agent, key, door, blue, green, red ,got_key, X, Y]
heistcolor: [agent, key, door, green, brown ,got_key, X, Y]  