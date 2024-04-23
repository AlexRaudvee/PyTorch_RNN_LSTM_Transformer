from Net import Generator

generate_names = Generator() # create the generator object with LSTM
generate_names.train() # train the LSTM with txt from config.py

def predict_word(start_: str = 'A'):
    top_generated = generate_names.generate(f'{start_}').split("\n")
    return top_generated[0]