import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from ..Bilinear_loss_old.BilinearLoss import BilinearLoss
from ..xsbert.models import XSRoberta, ReferenceTransformer
from ..xsbert.utils import plot_attributions_multi

# Load model
model_path = 'data/training_add2_nli_bert-base-uncased-2024-06-04_18-01-47_L0-9/eval/epoch9_step-1_sim_evaluation_add_matrix.pth'
sentence_transformer_model = SentenceTransformer('bert-base-uncased')
model_name = "training_add2_nli_bert-base-uncased_epoch9_"
bilinear_loss = BilinearLoss.load(model_path, sentence_transformer_model)

# Path for the Excel file
excel_path = f'figures/res_{model_name}.xlsx'
# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['texta', 'textb', 'A_sum', 'score', 'ra', 'rb', 'rr', 'sum_true', 'loss', 'true_label', 'pred_label'])


# Assume bilinear_loss is already initialized
transformer_layer = bilinear_loss.model[0]

# Save transformer layer to file
save_path = 'transformer_layerxx'
transformer_layer.save(save_path)
transformer = ReferenceTransformer.load(save_path)

pooling = bilinear_loss.model[1]
test_sbert = XSRoberta(modules=[transformer, pooling], sim_measure="bilinear", sim_mat=bilinear_loss.get_sim_mat())

print("Load model successfully.")
print(test_sbert)
print("Start to calculate.")

test_sbert.to(torch.device('cuda'))
test_sbert.reset_attribution()
test_sbert.init_attribution_to_layer(idx=10, N_steps=100)

# Multiple pairs of text
text_pairs = [
    ("Three dogs running through a field.", "The dogs are eating", "contradiction"),
    ("Three dogs running through a field.", "Three dogs are running in the daytime", "neutral"),
    ("Three workers, with life vests and hard hats, on a boat hanging from cables over water.", "Three people are on a boat.", "entailment"),
    ("Three workers, with life vests and hard hats, on a boat hanging from cables over water.", "Three lifeguards are hanging from the boat, trying to save the dog from drowning.", "neutral"),
    ("Three workers, with life vests and hard hats, on a boat hanging from cables over water.", "4 fisherman are hanging over the boat, trying to pet the shark", "contradiction"),
    ("A man wearing a short-sleeved blue shirt and carrying a blue backpack while using snow walking sticks treks through the snow with a woman wearing a long-sleeved blue shirt and black pants also using snow walking sticks.", "two people go to work in a blizzard", "neutral"),
    ("A man wearing a short-sleeved blue shirt and carrying a blue backpack while using snow walking sticks treks through the snow with a woman wearing a long-sleeved blue shirt and black pants also using snow walking sticks.", "a guy with a blue top carries a pack and makes his way through snow", "entailment"),
    ("A man wearing a short-sleeved blue shirt and carrying a blue backpack while using snow walking sticks treks through the snow with a woman wearing a long-sleeved blue shirt and black pants also using snow walking sticks.", "two men play with a snowman", "contradiction"),
    ("A man in a plaid shirt looking through a telescope lens.", "the man is watching the stars", "neutral"),
    ("A man in a plaid shirt looking through a telescope lens.", "the man is wearing a black shirt", "contradiction"),
    ("A man in a plaid shirt looking through a telescope lens.", "a man is looking through a telescope", "entailment"),
    ("A female runner dressed in blue athletic wear is running in a competition, while spectators line the street.", "There are children in the streets.", "contradiction"),
    ("A female runner dressed in blue athletic wear is running in a competition, while spectators line the street.", "There are people on the street.", "entailment"),
    ("A female runner dressed in blue athletic wear is running in a competition, while spectators line the street.", "The streets are empty.", "contradiction"),
    ("A man is painting a picture outside behind a crowd.", "A painter is creating a picture.", "entailment"),
    ("A man is painting a picture outside behind a crowd.", "A man is recreating the Mona Lisa outside in front of a crowd.", "neutral"),
    ("A young man in a blue hoodie doing a flip off of a half-wall that is covered in graffiti.", "A man does a flip off a wall.", "entailment"),
    ("A young man in a blue hoodie doing a flip off of a half-wall that is covered in graffiti.", "A young man does a flip off a wall and wears baggie clothes.", "neutral"),
    ("A young man in a blue hoodie doing a flip off of a half-wall that is covered in graffiti.", "A young man in a blue hoodie falls off a half-wall that is painted nicely.", "contradiction"),
    ("Two dogs run together near the leaves.", "Two dogs are running.", "entailment"),
    ("Two dogs run together near the leaves.", "The two dogs are sleeping on the pile of leaves.", "contradiction"),
    ("Two dogs run together near the leaves.", "The two dogs are running to play in the leaves.", "neutral"),
    ("A man with no shirt on is performing with a baton.", "A man is doing things with a baton.", "entailment"),
    ("A man with no shirt on is performing with a baton.", "A man throws a banana in the air.", "contradiction"),
    ("A man with no shirt on is performing with a baton.", "A man is trying his best at the national championship of baton.", "neutral"),
    ("A crowd wearing orange cheering for their team in a stadium.", "People wearing orange.", "entailment"),
    ("A crowd wearing orange cheering for their team in a stadium.", "A group of drag queens walking down the street.", "contradiction"),
    ("A crowd wearing orange cheering for their team in a stadium.", "Fans cheering on their team at the big game.", "entailment"),
    ("Female gymnasts warm up before a competition.", "Gymnasts get ready for a competition.", "entailment"),
    ("Female gymnasts warm up before a competition.", "Football players practice.", "contradiction"),
    ("Female gymnasts warm up before a competition.", "Gymnasts get ready for the biggest competition of their life.", "neutral"),
    ("Dark-haired man wearing a watch and oven mitt about to cook some meat in the kitchen.", "A man is going to surprise his wife with dinner.", "neutral"),
    ("Dark-haired man wearing a watch and oven mitt about to cook some meat in the kitchen.", "A man is cooking something to eat.", "entailment"),
    ("Dark-haired man wearing a watch and oven mitt about to cook some meat in the kitchen.", "A man is sitting watching TV.", "contradiction"),
    ("A worker peers out from atop a building under construction.", "A person is atop of a building.", "entailment"),
    ("A worker peers out from atop a building under construction.", "The unemployed person is near a building.", "contradiction"),
    ("A worker peers out from atop a building under construction.", "A man is atop of a building.", "neutral"),
    ("A young child joyfully pulls colorful tissue paper from a decorated box, looking for his present.", "a child pulls colorful tissue paper from a fancy box", "entailment"),
    ("A young child joyfully pulls colorful tissue paper from a decorated box, looking for his present.", "a child fights a bag, the bag is winning", "contradiction"),
    ("A young child joyfully pulls colorful tissue paper from a decorated box, looking for his present.", "a child opens a present on his birthday", "neutral"),
    ("A young boy in red leaping into sand at a playground.", "A young boy jumps onto his friends sand castle at a playground.", "neutral"),
    ("A young boy in red leaping into sand at a playground.", "A child is playing in the sand.", "entailment"),
    ("A young boy in red leaping into sand at a playground.", "A child does a cannonball into a pool.", "contradiction"),
    ("two boys reading superhero books", "Two boys reading a book about spiderman.", "neutral"),
    ("two boys reading superhero books", "Two boys watching a superhero show.", "contradiction"),
    ("two boys reading superhero books", "Two boys reading a piece of literature.", "entailment"),
    ("A man in a suit driving a horse-drawn buggy down a stone street.", "The man is driving a limosine.", "contradiction"),
    ("A man in a suit driving a horse-drawn buggy down a stone street.", "The man is driving a buggy.", "entailment"),
    ("A man in a suit driving a horse-drawn buggy down a stone street.", "The man is an amish man.", "neutral"),
    ("A blond little boy in an orange sweatshirt with red sleeves is using scissors to cut something.", "A little male has clothes on with a pair of scissors in his hands.", "entailment"),
    ("A blond little boy in an orange sweatshirt with red sleeves is using scissors to cut something.", "A blond little boy is cutting out little stars for his little sister.", "neutral"),
    ("A blond little boy in an orange sweatshirt with red sleeves is using scissors to cut something.", "A blond little boy is sleeping.", "contradiction"),
    ("A person dressed in white and black winter clothing leaps a narrow, water-filled ditch from one frost-covered field to another, where a female dressed in black coat and pants awaits.", "The man is dressed in summer clothing.", "contradiction"),
    ("A person dressed in white and black winter clothing leaps a narrow, water-filled ditch from one frost-covered field to another, where a female dressed in black coat and pants awaits.", "The female awaiting the man leaping is his sister.", "neutral"),
    ("A person dressed in white and black winter clothing leaps a narrow, water-filled ditch from one frost-covered field to another, where a female dressed in black coat and pants awaits.", "The man in white and black is leaping.", "entailment"),
    ("Elderly woman in blue apron balances a basket on her head on a sidewalk while talking to a woman dressed in black.", "The woman balancing a basket on her head is heading to her neighbors house.", "neutral"),
    ("Elderly woman in blue apron balances a basket on her head on a sidewalk while talking to a woman dressed in black.", "Elderly woman is balancing something on her head while having a conversation.", "entailment"),
    ("Blond woman overlooking Seattle Space Needle scene.", "A tourist checking out Seattle.", "neutral"),
    ("Blond woman overlooking Seattle Space Needle scene.", "Someone taking in the space needle view.", "entailment"),
    ("Blond woman overlooking Seattle Space Needle scene.", "A man enjoying the view of the golden gate bridge.", "contradiction"),
    ("An old man looking over a sculpture.", "The man is young", "contradiction"),
    ("An old man looking over a sculpture.", "the man is at an art gallery", "neutral"),
    ("An old man looking over a sculpture.", "the man is old", "entailment"),
    ("The soccer team, clad in blue for the match, began to counter down the field in front of the defender, clad in red.", "The soccer team in blue plays soccer.", "entailment"),
    ("The soccer team, clad in blue for the match, began to counter down the field in front of the defender, clad in red.", "The blue soccer team is playing its first game of the season.", "neutral"),
    ("The soccer team, clad in blue for the match, began to counter down the field in front of the defender, clad in red.", "A soccer team hangs out in the locker room.", "contradiction"),
    ("Two little kids without shirts are sitting down facing each other.", "Two children are sleeping at daycare", "contradiction"),
    ("Two little kids without shirts are sitting down facing each other.", "Two children are sitting shirtless indoors", "neutral"),
    ("Two little kids without shirts are sitting down facing each other.", "Two young children are playing with each other and giggling while playing with markers", "neutral"),
    ("Workers standing on a lift.", "Workers walk off a lift", "contradiction"),
    ("Workers standing on a lift.", "Workers walk home from work", "contradiction"),
    ("Workers standing on a lift.", "Workers stand on a lift", "entailment"),
    ("Two people talking on a dock.", "fishermen at the dock", "neutral"),
    ("Two people talking on a dock.", "women shopping at the mall", "contradiction"),
    ("Two people talking on a dock.", "people outside", "entailment"),
    ("a young boy using a field microscope to identify a field specimen during a field trip.", "The boy is on a science field trip.", "neutral"),
    ("a young boy using a field microscope to identify a field specimen during a field trip.", "The young girl looks through the telescope.", "contradiction"),
    ("a young boy using a field microscope to identify a field specimen during a field trip.", "The boy is looking through a microscope.", "entailment"),
    ("A man has flung himself over a pole with people and canopies in the background.", "A guy has hurled himself in the air.", "entailment"),
    ("A man has flung himself over a pole with people and canopies in the background.", "The man is running after a robber.", "contradiction"),
    ("A man has flung himself over a pole with people and canopies in the background.", "A person is performing a trick at the circus.", "neutral"),
    ("A golfer is getting ready to putt on the green, with a crowd of people watching in the background.", "A golfer readies to putt the ball.", "entailment"),
    ("A golfer is getting ready to putt on the green, with a crowd of people watching in the background.", "The golfer is getting ready to putt on the green, with a crowd watching in the background.", "entailment"),
    ("A golfer is getting ready to putt on the green, with a crowd of people watching in the background.", "The golfer retired from play today.", "contradiction"),
    ("A child in a ninja outfit does a jumping kick.", "a child is wrestling with bears", "contradiction"),
    ("A child in a ninja outfit does a jumping kick.", "a child does a jumping kick", "entailment"),
    ("A child in a ninja outfit does a jumping kick.", "a child in a black ninja suit does a kick", "neutral"),
    ("Three men are sitting outside on chairs with red seats.", "Men are having a conversation outside.", "neutral"),
    ("Three men are sitting outside on chairs with red seats.", "Three men are sitting at the kitchen table.", "contradiction"),
    ("Three men are sitting outside on chairs with red seats.", "Men are sitting outside.", "entailment"),
    ("A person is sitting in front of a graffiti covered wall.", "There's a place to sit near a wall", "entailment"),
    ("A person is sitting in front of a graffiti covered wall.", "A person is sitting outside", "neutral"),
    ("A person is sitting in front of a graffiti covered wall.", "A person is laying at home", "contradiction"),
    ("A man cooking over high flames.", "A man is raking the yard.", "contradiction"),
    ("A man cooking over high flames.", "A man is cooking for his friends.", "neutral"),
    ("A man cooking over high flames.", "A person is preparing some food.", "entailment"),
    ("A man cooking with fire in like 5 pots at the same time!", "A man is cooking with a lot of pots.", "entailment"),
    ("A man cooking with fire in like 5 pots at the same time!", "A man is cooking dinner for his family in a bunch of pots.", "neutral"),
    ("A man cooking with fire in like 5 pots at the same time!", "A man is cooking with a bunch of ovens.", "contradiction"),
    ("A man with a long white beard is examining a camera and another man with a black shirt is in the background.", "a man is with another man", "entailment"),
    ("A man with a long white beard is examining a camera and another man with a black shirt is in the background.", "A man takes a picture of a woman", "contradiction"),
    ("A man with a long white beard is examining a camera and another man with a black shirt is in the background.", "A man is with a cowboy", "neutral"),
    ("Four women are taking a walk down an icy road.", "The road is dangerous for the four women to try to walk on because it is covered in ice.", "neutral"),
    ("Four women are taking a walk down an icy road.", "The women are walking on the ice.", "entailment"),
    ("Four women are taking a walk down an icy road.", "Four women are walking near the dry highway.", "contradiction"),
    ("Four men stand in a circle facing each other playing brass instruments which people watch them.", "People love the music", "neutral"),
    ("Four men stand in a circle facing each other playing brass instruments which people watch them.", "The men are watching a sports channel", "contradiction"),
    ("Four men stand in a circle facing each other playing brass instruments which people watch them.", "The men are playing music", "entailment"),
    ("A man is shooting a gun outdoors, on what looks like a beautiful sunny day.", "A man is shooting a bow and arrow on a rainy day.", "contradiction"),
    ("A man is shooting a gun outdoors, on what looks like a beautiful sunny day.", "A man is shooting a gun at targets on a nice day.", "entailment"),
    ("A man is shooting a gun outdoors, on what looks like a beautiful sunny day.", "A man is shooting a gun outside with his friends.", "neutral"),
    ("A group of football players is standing behind a coaching official.", "There is only one person present.", "contradiction"),
    ("A group of football players is standing behind a coaching official.", "They are college football players.", "neutral"),
    ("A group of football players is standing behind a coaching official.", "There are multiple people present.", "entailment"),
    ("Five young men are in a loft, with one holding a whiteboard and one making a shaka sign in front of the television.", "Five men watch tv.", "neutral"),
    ("Five young men are in a loft, with one holding a whiteboard and one making a shaka sign in front of the television.", "Some guys are in a living room.", "neutral"),
    ("Five young men are in a loft, with one holding a whiteboard and one making a shaka sign in front of the television.", "Some people are at work.", "contradiction"),
    ("A man in a black ball cap, black jacket and pants sitting in front of a building painted white and blue with the words ING and PEOPLE written on it.", "The man is sitting in front of a school.", "neutral"),
    ("A man in a black ball cap, black jacket and pants sitting in front of a building painted white and blue with the words ING and PEOPLE written on it.", "The man is standing in front of a building.", "contradiction"),
    ("A man in a black ball cap, black jacket and pants sitting in front of a building painted white and blue with the words ING and PEOPLE written on it.", "The man is in front of a building.", "entailment"),
    ("People seated at long tables all facing the same direction, some writing and some watching.", "A group of people are sitting at tables.", "entailment"),
    ("People seated at long tables all facing the same direction, some writing and some watching.", "A group of people are sitting at the tables sharing a meal.", "contradiction"),
    ("People seated at long tables all facing the same direction, some writing and some watching.", "A family reunion is in progress as relatives sit at tables and write down each other's addresses.", "neutral"),
    ("An elderly woman is preparing food in the kitchen.", "A person makes dinner.", "neutral"),
    ("An elderly woman is preparing food in the kitchen.", "A man cleans the kitchen.", "contradiction"),
    ("An elderly woman is preparing food in the kitchen.", "A woman makes food.", "entailment"),
    ("Four women competitively rollerskating around an area.", "Four women enjoying rollerskating.", "neutral"),
    ("Four women competitively rollerskating around an area.", "Women rollerskating around an area", "entailment"),
    ("Four women competitively rollerskating around an area.", "four men are rollerskating.", "contradiction"),
    ("A group of cleaners are sweeping up animal feces from the street during a parade or festival.", "A man walking on water.", "contradiction"),
    ("A group of cleaners are sweeping up animal feces from the street during a parade or festival.", "Workers cleaning up after St Patrick's Day.", "neutral"),
    ("A group of cleaners are sweeping up animal feces from the street during a parade or festival.", "A group of cleaners after a parade.", "entailment"),
    ("A young girl in a pink shirt playing with her barbie.", "The young girl is having fun.", "neutral"),
    ("A young girl in a pink shirt playing with her barbie.", "The young girl is playing with a race car.", "contradiction"),
    ("A young girl in a pink shirt playing with her barbie.", "The young girl is playing with a toy.", "entailment"),
    ("A car is loaded with items on the top.", "The car has stuff on top.", "entailment"),
    ("A car is loaded with items on the top.", "The car is going on a trip.", "neutral"),
    ("A car is loaded with items on the top.", "The car is a convertible.", "contradiction"),
    ("A man wearing a snorkel and goggles gives a thumbs up as he and another person speed through the water.", "There are two guys above the water.", "neutral"),
    ("A man wearing a snorkel and goggles gives a thumbs up as he and another person speed through the water.", "Two guys are on a lake.", "neutral"),
    ("Several women wearing dresses dance in the forest.", "There are men dancing", "contradiction"),
    ("Several women wearing dresses dance in the forest.", "The women are older", "neutral"),
    ("Several women wearing dresses dance in the forest.", "there are several women", "entailment"),
    ("A boy in a green shirt on a skateboard on a stone wall with graffiti", "A long-haired boy riding his skateboard at a fast pace over a stone wall with graffiti.", "neutral"),
    ("A boy in a green shirt on a skateboard on a stone wall with graffiti", "A boy riding a skateboard on a stone wall.", "entailment"),
    ("A boy in a green shirt on a skateboard on a stone wall with graffiti", "A boy in a green shirt rollerblading through the tunnel.", "contradiction"),
    ("The back of a woman wearing a white jacket with blue jeans walking towards a flock of birds.", "Someone person was near a bunch of birds.", "entailment"),
    ("The back of a woman wearing a white jacket with blue jeans walking towards a flock of birds.", "There was a dog inside.", "contradiction"),
    ("The back of a woman wearing a white jacket with blue jeans walking towards a flock of birds.", "There were some birds near a woman.", "entailment"),
    ("Dog running with pet toy being chased by another dog.", "A dog is being chased by a cat", "contradiction"),
    ("Dog running with pet toy being chased by another dog.", "a black dog with a toy is being based by a brown dog", "neutral"),
    ("Dog running with pet toy being chased by another dog.", "dog is running and being chased by another dog", "entailment"),
    ("Two children and a woman climb up the stairs on a metal electric pole-like structure.", "A woman and her kids climb up the stairs", "neutral"),
    ("Two children and a woman climb up the stairs on a metal electric pole-like structure.", "A woman and two children climb up the stairs", "entailment"),
    ("A man is sitting on the floor, sleeping.", "A man is lying down, sleeping.", "contradiction"),
    ("A man is sitting on the floor, sleeping.", "A man is sleeping on the floor.", "entailment"),
    ("A man is sitting on the floor, sleeping.", "A young man is sleeping while in the sitting position.", "entailment"),
    ("A woman, wearing a colorful bikini, rests laying down next to the blue water.", "The woman is wearing a prom dress.", "contradiction"),
    ("A woman, wearing a colorful bikini, rests laying down next to the blue water.", "The woman is a super model.", "neutral"),
    ("A woman, wearing a colorful bikini, rests laying down next to the blue water.", "A woman in a bikini lays near the water.", "entailment"),
    ("A brown dog jumps over an obstacle.", "The dog is a chihuahua.", "neutral"),
    ("A brown dog jumps over an obstacle.", "The dog is in a pool.", "contradiction"),
    ("A brown dog jumps over an obstacle.", "The dog is in the air.", "entailment"),
    ("A brown dog carries an object in its mouth on a snowy hillside.", "A dog is taking something to its owner.", "neutral"),
    ("A brown dog carries an object in its mouth on a snowy hillside.", "A dog is carrying something.", "entailment"),
    ("A brown dog carries an object in its mouth on a snowy hillside.", "A dog is carrying an object down a grassy side.", "contradiction"),
    ("A group of people watch a breakdancer in a red jacket do a one-handed trick.", "A person is doing tricks in front of crowd.", "entailment"),
    ("A group of people watch a breakdancer in a red jacket do a one-handed trick.", "Tricks are made by a person in a red jacket while people watching him.", "entailment"),
    ("A group of people watch a breakdancer in a red jacket do a one-handed trick.", "Man who wears red jacket got an accident.", "contradiction"),
    ("Two men in neon yellow shirts busily sawing a log in half.", "Two men are cutting wood to build a table.", "neutral"),
    ("Two men in neon yellow shirts busily sawing a log in half.", "Two men are hammering.", "contradiction"),
    ("Two men in neon yellow shirts busily sawing a log in half.", "Two men are using a saw.", "entailment"),
    ("Two strong men work to saw a log.", "two strong men are working", "entailment"),
    ("Two strong men work to saw a log.", "two strong men are having a beer", "contradiction"),
    ("Two strong men work to saw a log.", "two strong men work to saw the oak log", "neutral"),
    ("A black man wearing a red belt and a white man wearing a blue belt are pictured in the act of practicing martial arts inside a building with a woman looking on in the background.", "The men are novices at martial arts.", "neutral"),
    ("A black man wearing a red belt and a white man wearing a blue belt are pictured in the act of practicing martial arts inside a building with a woman looking on in the background.", "The men are blackbelts.", "contradiction"),
    ("A black man wearing a red belt and a white man wearing a blue belt are pictured in the act of practicing martial arts inside a building with a woman looking on in the background.", "A man is wearing a red belt.", "entailment"),
]

true_labels = ['contradiction', 'entailment', 'neutral']  # Add corresponding true labels

for i, (texta, textb, label) in enumerate(text_pairs):
    if i > 100: 
        break
    print(f"Processing pair {i+1}:")
    print(f"texta: {texta}")
    print(f"textb: {textb}")
    print(f"label: {label}")
    
    A, tokens_a, tokens_b, score, ra, rb, rr = test_sbert.explain_similarity_gen(
        texta, 
        textb, 
        move_to_cpu=False,
        return_lhs_terms=True
    )

    #print("A.shape: ", A.shape)

    A_sum = A.sum(dim=(1, 2)).cpu().numpy()
    score_np = score.cpu().numpy()
    ra_np = ra.cpu().numpy()
    rb_np = rb.cpu().numpy()
    rr_np = rr.cpu().numpy()

    sum_true = score_np - ra_np - rb_np + rr_np
    loss = A_sum - sum_true

    pred_label = sum_true.argmax()

    result_values = f"A_sum_{A_sum}__sum_{sum_true}__loss_{loss}_label{label}_"

    fig_save_path = f"figures/fig_{model_name}_pair_{i+1}_{result_values}.png"
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)

    plot_attributions_multi(
        A, 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        show_colorbar=True, 
        shrink_colorbar=.5,
        dst_path=fig_save_path,
        labels=["Contradiction", "Entailment", "Neutral"]
    )

    print(f"Saved figure to {fig_save_path}")
    print(A_sum, sum_true)

    # Append results to DataFrame
    results_df = results_df.append({
        'texta': texta,
        'textb': textb,
        'A_sum': A_sum.tolist(),
        'score': score_np.tolist(),
        'ra': ra_np.tolist(),
        'rb': rb_np.tolist(),
        'rr': rr_np.tolist(),
        'sum_true': sum_true.tolist(),
        'loss': loss.tolist(),
        'true_label': true_labels[i],
        'pred_label': pred_label
    }, ignore_index=True)

    # Save DataFrame to Excel after each iteration
    results_df.to_excel(excel_path, index=False)

print("All pairs processed.")
