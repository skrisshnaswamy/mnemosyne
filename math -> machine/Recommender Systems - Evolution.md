### Part 1 — The early days: simple, but powerful

Think of recommender systems starting with very human-style intuition.

#### **Collaborative Filtering (CF).**  
This is where everything really began. The idea: _people who behave like you might like the things you like_.  
No fancy math. Just patterns in user–item interactions.

Core idea: Instead of looking at the thing in question, look at the people.

> [!TIP] Example
If _User A_ watched _Toy Story_ and _Finding Nemo_
and if _User B_ watched _Toy Story_, _Finding Nemo_ and _Monsters Inc._
Then _User A_ will likely watch _Monsters Inc._ too.

But CF struggled when there wasn’t enough data (cold start).

---

### Part 2 — The math gets deeper: discovering hidden dimensions

#### **Matrix Factorization (MF).**  
This truly changed the field after the Netflix Prize (around 2006–2009). 
MF says: _users and items secretly live in a low-dimensional world of “latent factors”_.  
This was the first “big leap” from simple heuristics to learned representations.

So what exactly is MF?
You see, the above explanation of CF looks very verbose and if we were to represent it mathematically, it would look like a gaint matrix with -
- **Rows** are the Users.
- **Columns** are the Movies.
- **Cells** are the ratings (like 1 to 5 stars).

But then most of the entries in this matrix would be 0 - because nobody has watched every single movie!

So MF brought in the **Hidden traits** concept. Mathematically, this is simply *dimensionality reduction* and the lower dimension is called the **latent space**.
Instead of storing the actual ratings, MF tells you if you were to break down this giant matrix into 2 smaller matrix of hidden traits - you avoid:
- **Storage:** Storing billions of zeros which is a waste of memory.
- **Search:** Doing a 2D search (finding "Who is most like me?") across millions of people, which is computationally expensive ($O(N^2)$ or worse).
But what hidden traits btw?
Think of it as
- **User Table:** traits that describe what each user likes (e.g., "Likes Action," "Likes Romance").
- **Movie Table:** traits that describe what each movie is (e.g., "Is Action," "Is Romance").

But these traits are **not** handcrafted. And how many traits is also something not exactly well defined. (It's a hyperparam you can tweak.)
So
- **The User Matrix:** (#Users × $k$ traits)
- **The Item Matrix:** (#Items × $k$ traits)

The number $k$ is the **dimension** (or "rank"). Usually, $k$ is small (like 50 or 128). Multiplying a row from the User matrix by a column from the Item matrix is a **dot product**, which is incredibly fast for computers to calculate.1

But why are they not handcrafted?
You see,  it feels natural to explicitly define what a movie or a user “is” using human-understandable features like genres, actors, era, or directors.

If we did this, each movie would be represented as a mostly **binary or sparse vector**:
- A movie is either Comedy or not
- A movie is either Sci-Fi or not

This works, but it comes with a strong assumption:
> We assume _we already know_ the most important dimensions along which users make decisions.

Matrix Factorization deliberately avoids this assumption.

Instead of explicitly defining features, MF **infers them directly from user–item interaction data**.  
The only goal is to find a low-dimensional representation that best explains observed ratings.

So when the computer or the model is implicitly learning it, it could actually capture traits like "Movies with orange posters" if it turns out a specific group of people really loves those, or "Movies that are exactly 90 minutes long." or say "Dark 80s Sci-fi". Which would result in a continuous number dense vectors.

>[!SUCCESS] Core idea
Handcrafted features reflect **human intuition**.
Latent factors reflect **statistical regularities in behavior**.
> 
> MF learned _what matters_ directly from data, instead of being told.


---

### Part 3 — Neural recommenders enter the chat

#### Neural Matrix Factorization (NeuMF)
So in MF, we used linear way of combining users and items - dot product / linear projection. But in real world, human behaviour is hardly linear.

So in 2017, people asked **what if we replaced the dot product with a deep neural network?**
>[!ABSTRACT] :arxiv: \[1708.05031\] Neural Collaborative Filtering by _Xiangnan He_
>[![Arxiv: NeuMF](https://velog.velcdn.com/images/gyuu_katsu/post/964a3e47-b9d2-4343-a4ba-2d978ae7f7d7/image.png)](https://arxiv.org/abs/1708.05031)

They architected the NCF as a 2-track system where the model looks at users and entities (movies) with **2** different lenses _at the same time_.
**Track 1:** This is the **linear** lens - called **Generalized Matrix Factorization (GMF)**
This behaves like a typical MF - does element wise multiplication i.e. dot product. And this lens is good at picking up simple relationships

**Track 2:** This is the **deep** lens
This is a standard deep neural network. Instead of multiplying the user and item, it **concatenates** them (sticks them together side-by-side into one long vector).
- It then passes this long vector through several hidden layers.
- Because it has "brains" (non-linear activation functions like ReLU), it can find weird, complex "if-then" patterns that a simple dot product would miss.
> **Example:** It might learn that "User A likes Horror movies _only_ if they were made in the UK _and_ are under 90 minutes long."

At the very end, the model combines the outputs from both tracks to make one final prediction: **"Will this user click?"** 🖱️

>[!SUCCESS] Core idea
>So **NeuMF** is a hybrid model - it uses the **linear** track to keep the strengths of the classic MF, but then it augments it with the **deep** track to learn complex non-linear human quirks

---
#### Wide & Deep Models
This was the model that Google made famous for the Google Play Store.
Introduced in the 
> [Wide and Deep learning in Recommender systems](https://arxiv.org/abs/1606.07792)
> [![**"Wide & Deep Learning for Recommender Systems"**](https://i.ytimg.com/vi/gL8jwhlb3xY/hqdefault.jpg)](https://arxiv.org/abs/1606.07792) 
> paper by [**Cheng et al. (2016)**](https://arxiv.org/abs/1606.07792)

If **NeuMF** was about combining "Linear" and "Deep" to improve MF, **Wide & Deep** is about balancing two human cognitive abilities: **Memorization** and **Generalization**.

The Google engineers while creating a model for their playstore recommendation, noticed a problem:
- Purely "Wide" models (like basic logic) were great at **Memorization** but failed to suggest new things - i.e. too rigid
- Purely "Deep" models (like Neural Networks) were great at **Generalization** but sometimes ignored very obvious, specific rules - i.e. too creative

**Summary:** Wide & Deep is like a brain that has a "Memory" for specific rules (Wide) and an "Imagination" for similar things (Deep).

![[widendeep_arch.png]]

So this arch, has:
**A "Wide" Part (Memorization)**
Think of this as the system "memorizing" specific, rules.
- **Example:** People who search for "iced latte" almost always want "iced latte."
- This part of the model uses a simple linear layer. It’s great at remembering very specific combinations (like "User X + App Y") that have worked well in the past.
- It memorizes that **"Users who search for 'chicken' often order 'fried chicken'."** * It uses "cross-product features" (basically `AND` rules). If a user is in `Category A` AND searching for `Item B`, recommend `C`. It’s very precise and great for "niche" items.

**A "Deep" Part (Generalization)**
Think of this as the system "generalizing" or "exploring" new things.
- **Example:** If you like "iced lattes," you might also like "cold brew" or "frappuccinos."
- This is a Deep Neural Network that looks at the **similarities** between items. Even if you've never ordered a cold brew before, the deep part sees it's similar to things you _do_ like.
- This part **uses embeddings** (those "hidden trait" vectors we talked about).
- It learns that **"Fried chicken"** is similar to **"Burgers"** because they are both "Fast Food."
- Even if you've never ordered a burger, the Deep part sees the similarity and suggests one. It "generalizes" your taste.

By **jointly training** them, the model uses the Wide side to handle the "rules and exceptions" and the Deep side to handle "similarity and variety."

>[!ABSTRACT] ▶️ Wide & Deep Learning: Memorization + Generalization with TensorFlow (TensorFlow Dev Summit 2017)
>[![## Wide & Deep Learning: Memorization + Generalization with TensorFlow (TensorFlow Dev Summit 2017)](https://img.youtube.com/vi/NV1tkZ9Lq48/0.jpg)](https://www.youtube.com/watch?v=NV1tkZ9Lq48)

---
#### Deep structured models
YouTube 2016 - **DNN Recommenders**
Introduced in the paper [**Deep Neural Networks for YouTube Recommendations"**](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) by [**Covington et al. (2016)**](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf). 📽️

So when YouTube engineers sat down to design their system, they faced a unique problem: **Scale.** They have billions of videos and billions of users. 🤯 They couldn't just run a complex deep-learning calculation for every single video every time you refresh your homepage.

So, they split the problem into two distinct "acts":

![[youtube_dnn_rec_1.png]]
##### Step 1: Candidate Generation (The Funnel) 🌪️
Think of this as a "quick and dirty" filter.
- **Goal:** Turn millions of videos into a few hundred.    
- **How:** It looks at your recent history (what you watched, what you searched) and uses a simple neural network to grab a broad "bucket" of videos you might like.
- It doesn't have to be perfect; it just has to be fast.
##### Step 2: Ranking (The Judge) ⚖️
Now that we only have ~500 videos to worry about, we can afford to be "smart."
- **Goal:** Arrange those 500 videos in the exact order of how likely you are to watch them.
- **How:** This part uses a much more complex neural network. It looks at "fine-grained" details:
    - Is the thumbnail attractive to you?
    - Have you ignored this video three times already today?
    - How fresh is this video?

In essence, the candidate generation treats the massive recommendation problem as a massive classification problem - _"Given everything we know about this user, which video are they most likely to watch next?"_
And the next ranking stage, actually assigns a score to each video based on the **expected watch time**. YouTube prefers videos you'll actually spend time on, not just clickbait.
![[youtube_dnn_rec_2.png]]


---

### Part 3.5 - Two-Tower Architectures / Dual Encoder Models

This is the idea:  
_One tower encodes the user; another encodes the item; match them in an embedding space._
This became extremely popular because it scales beautifully for retrieval.

In the previous YouTube DNN Recommender, the Candidate Generation stage seems like a really heavy task. How do we speed that up? The **Two-Tower** architectures is the answer. They are the modern standard for the "Candidate Generation" (Stage 1).

So instead of mixing wide and deep features or using a combination of the user's and item's latent features, it tries to first represent the users and items in the latent space and then **only at the last step** combines these 2. By decoupling them until the last step, it can now define different cadence and schedule for learning those latent features in isolation.

**⚡ Speed Hack (pre-computation):** For example, it can actually learn the video's latent representation offline, at the time of the video upload. And the user's vector can be learned at the time when it's generating the recommendation.
>- They can calculate the vector for a video the moment it's uploaded and store it in a giant "vector library" (a **Vector Database**).
>- So when you open the app, the system only has to run the **User Tower** once to get your vector.
>- Then, it just compares your one vector against the millions of pre-calculated video vectors. This is like having a "key" and checking it against a million "locks" simultaneously.

>[!INFO] Vector Databases
>While **RAG** (Retrieval-Augmented Generation) definitely made Vector Databases a household name in AI recently, the "Two-Tower" architecture used in recommendation systems was actually using these same principles years before LLMs went mainstream. 🕰️
>
In the recommender world, we call this **MIPS** (Maximum Inner Product Search). It’s essentially the same "retrieval" step as RAG:
>- **Recommendation:** User Vector ➡️ Search Vector DB ➡️ Get top Items. 🛍️    
>- **RAG:** Question Vector ➡️ Search Vector DB ➡️ Get top Document chunks. 📄

This is what it means by building the 2 separate towers. 

How it works:
1. **User tower:** This is a deep net which learns everything about you and represents you in a single vector
2. **Item (Video) tower:** This is also a deep net which learns everything about the item (video) and represents it in a single vector.
3. **The meeting:** This is the last step of the candidate generation step which **combines** the user and the item vectors via a simple **dot product** like the MF to see how well they match.

The decoupling of the 2 towers help make the models simple and faster. But it also means the model cannot learn really complex relationship between the items and users - like _"This user likes this specific actor only when they are in a comedy."_ So the next stage of ranking is complex enough to handle these complex interactions.

---

### Part 4 — The “deep retrieval era”: sequence models, semantics & embeddings
The Order Matters! 🎞️

Up until now, models looked at user or item history as a "bag of items" and not necessarily care about the order. But in real life, interests of users **changes over time**. 

Example:
1. You watch a **Cooking Tutorial** 🍳
2. Then you watch a **Knife Sharpening** video 🔪
3. The next thing you want is probably a **Cutting Board** review, not another recipe.

This is where Sequence Models come into the picture.
Now Sequence Models aren't necessarily Transformer based models. Any DL model arch. that looked at input as a sequence of tokens or actions is a Sequence Model.
So that would include:
- RNNs and GRUs - Models that had _memory_ that carried information from 1 item/token to next
- Transformers - The _attention_ based model, that could say "Out of the last 10 things this user did, their 2nd and 9th actions are the most important for predicting what they want now."

> [!TIP] Example
Imagine your click history is a sentence:
_"I bought a camera 📸, then a lens 🔍, then a tripod 🔭, now I need a..."_ 
>
> ---
A Sequence Model doesn't just see "camera, lens, tripod." It sees the **progression**. It predicts **"Camera Bag"** because that's the logical next step in your journey.

So the question moved from **What does this person like?** to **What is this person doing right now?**

#### Part 4.A - Modern Representation
So up until now, items and users were represented as **Random IDs** and the machine learned from scratch that movie 827 is infact a horror movie via how people interacted with it. So why not give this item a **name** instead that actually mean something, right from the start.

##### Semantic IDs

That's where **Semantic IDs** come into play. It's like encoding the **DNA of the item** itself. 
The **Semantic IDs** are like a barcode that helps understand what's inside these items.
	Like How it's made and what's it made of - content, description, category, etc.
This produces a meaningful ID that ensures when 2 items have _similar_ Semantic IDs, they probably belong to the same category or have similar content etc. 

But why is this helpful? It's very useful with **cold start** problem. Because now the machine doesn't need to learn about a new item or movie or user from scratch, but can rather have some similarity info baked into these semantic IDs which can inform that although these items are new they are of the same category for example, that we already learned about.

##### Knowledge Graphs

So if we were to consider semantic IDs like an array of numbers that tells me what the items are made of, Knowledge graph is like building a network of information that helps us classify the items in question more clearly. 

Semantic ID : Barcode :: Knowledge Graph : Social network

So you see, knowledge graph takes the whole concept of baking in meaning into IDs to a whole new level. It helps bake in the detail like why the item exists and where does it fit in, in a web of real world relationships.
![[kg_example.png]]

**Why this helps the Recommender:**
1. **Explainability:** Instead of just saying "You might like this," the system can say, "We’re suggesting _John Wick_ because you liked _The Matrix_ and they both star Keanu Reeves." 🗣️
2. **The "Bridge" Effect:** Even if no one in the history of the app has ever watched both _The Matrix_ and a specific indie action movie, the KG can "bridge" them together if they share the same stunt coordinator or filming style.
3. **Reasoning:** It allows the model to "walk" along the paths. If you've been watching many movies directed by a specific person, the model "walks" one step further on the graph to find their influences.

But both Semantic IDs and Knowledge Graphs feel like a lot of hand crafting being needed. Does it mean, we now move from letting the machine learn dense complex representations to more hand crafted representations? **Not exactly**

> [!INFO] **Semantic IDs: Automated, not Hand-Crafted 🤖**
In modern architectures like HSTU, TIGER etc., this work is delegated to a pre-trained text encoder like T5 or BERT.
The encoder consumes raw text like movie reviews or descriptions etc. and turns it into an embedding first. And then a special algorithm **RQ-VAE** squashes them into short codes i.e. Semantic IDs.
The machine _discovers_ the similarities itself. It might put _The Matrix_ and _Inception_ into the same "ID neighborhood" not because we told it they are "Sci-Fi," but because it analyzed thousands of words in their descriptions and found they are mathematically similar.

> [!INFO] **Knowledge Graphs: The LLM "Automated Researcher" 🕸️**
> Previously KGs were hand crafted and it took too long. But these days we use LLMs. LLMs read thousands of wiki pages and what not and automatically extract the facts into graphs.
> Systems like Amazon's **COSMO** even use LLMs to _guess_ common sense relationships. 

All of these together form what people call **“representation-rich recommendation.”**

---

### Part 5 — Generative recommendation: where everything converges

This is the cutting edge.

Instead of predicting “the next item embedding,” generative recommenders:

• model the sequence of user actions as text-like tokens
• generate the next token (item) autoregressively
• sometimes unify retrieval + ranking in one generative framework
• use large language models (LLMs or smaller domain-specific generative models)


> [!TIP] So, **What makes a model "Generative"?**
A recommender becomes "Generative" when it stops matching your vector to a database and starts **decoding** or "typing out" the answer.

#### Advanced Architectures: TIGER and HSTU 🐯

**TIGER (Generative Retrieval):** 🐅
- This treats recommendation like a **language problem**.
- TIGER introduced the idea that items should be represented as a sequence of tokens (Semantic IDs) that the model can "speak." Instead of one big random number, the model generates the item token-by-token (e.g., `Category: Electronics` ➡️ `Sub-category: Phone` ➡️ `Brand: Apple`). 🐯
- Instead of "searching" a database, the model **predicts the Semantic ID** of the next item you want, almost like it's "typing" the name of the next product.
- It’s like the model is saying, _"Based on your history, I'm going to type out the ID for a 'Blue Leather Hiking Boot'."_
- And because the IDs are semantically linked, if the model "types" a slightly wrong ID, it usually still lands on an item that is very similar to what you wanted!
- Semantic ID allows models like **TIGER** to "predict" your next item by its name.

**HSTU (Hierarchical Sequential Transformer User):** 🏗️
- This is a "Super-Transformer" designed specifically for the massive scale of recommenders.
- It's built to handle **extremely long histories** (thousands of clicks) without slowing down, making it much more efficient than the standard Transformers used in ChatGPT.
	- Standard Transformers (like the ones in ChatGPT) get very slow if you have a huge history (thousands of clicks).
	- HSTU uses clever math to handle **massive sequences** much more efficiently than a normal Transformer, making it possible to use these generative ideas in giant, real-time apps.
- Semantic ID allows **HSTU** to remember your history more efficiently.

These are foundational papers that enabled Generative Recommendations. We can think of these as native RecSys.

The idea is:  
_recommendation becomes a sequence generation problem rather than a scoring problem._

This is the direction food-ordering, e-commerce, and entertainment platforms are actively exploring.


But these days researchers are looking at how to teach LLMs to recommend. 
This includes:
• **Google’s PaLM-based RecSys**  
• **Meta’s LLM-Rec**  
• **Alibaba’s ACE, GENREC**  
• **Amazon’s GPT-Rec-style prototypes**

These LLMs already have massive world knowledge and deep reasoning at their disposal. 

---

### Part 6 - How the tokenization is done in Generative Models? 🏷️

In a normal LLMs, words are broken into small pieces called **tokens**. For a generative recommender to work, it has to treat **products** just like words.

So, the question is: How do you turn a specific pizza 🍕 or a movie 🎬 into a sequence of tokens that a model can "speak"?
This process is called **Semantic Tokenization**.
Think of it like, instead of one giant ID, we break the item into a "code" (like a library Dewey Decimal number).

#### Heirarchical approach
Imagine a code like `12-45-78`.
- `12` might mean "Italian Food"
- `45` might mean "Main Dish"
- `78` might mean "Spicy"

#### RQ-VAE - Vector Quantization
This is a model that takes a deep, complex embedding (a list of numbers) and squashes it into these discrete steps (like the `12-45-78` above)


### Part 7 - How Generative Models predict the next item? 🔮

**Autoregressive Generation:** The model looks at your history and starts "typing" the code for the next item.
**The "Beam Search" Advantage:** The model doesn't just guess one item. It explores multiple "paths" (sequences of tokens) to find the most likely next item. Because the tokens are semantic, if it starts with `12` (Italian), it's already narrow its world down to exactly what you're in the mood for.

> [!INFO] 🔎 BEAM SEARCH
> It’s the decision-making "eye" of the model!
> Beam Search is like having a small team of researchers explore the most promising paths simultaneously, rather than just one person running blindly ahead.
> 
> Imagine you are trying to guess a word, letter by letter.
> - **Greedy Search:** You always pick the most likely next letter. If you start with "T," and the most likely next letter is "h," you pick "h." You never look back. 🏃‍♂️
> - **Beam Search:** You keep a "beam" of several top options at once. Instead of just "Th," you might keep "Th," "To," and "Tr." 🌲

#### Why TIGER uses Beam Search 🐅
In the **TIGER** paper, the model generates an item's **Semantic ID** (like `12-45-78`) token by token.
1. **Step 1:** The model predicts the first token. If your **Beam Width** is 3, it keeps the top 3 possibilities (e.g., `12`, `15`, `09`).
2. **Step 2:** For _each_ of those 3, it predicts the second token. Now it has many combinations! It calculates the probability of each full path (e.g., `12-45` vs. `12-30`) and again keeps only the top 3 overall.
	1. **Search Bias** / **Pruning Bias** - It's the risk that because we "cut off" certain paths early, we might lose a sequence that started weak but would have ended up being the perfect recommendation.
3. **The Result:** This prevents the model from making a "short-sighted" mistake. A first token might look very likely, but lead to a dead end. Beam Search allows the model to find the **best overall ID** even if the first part of the ID wasn't the absolute #1 favorite.

##### 1. The "Narrow Path" Problem 🛤️
If the top 3 first tokens are all from the "Action Movie" category, the beam search is now **biased** toward only recommending action movies. Even if there was a "Comedy" token that was the 4th most likely, it gets deleted forever.

This is a trade-off:
- **The Pro:** It makes the system incredibly fast. Instead of checking millions of items, we only check a tiny fraction. ⚡
- **The Con:** We might miss a "hidden gem" that has a slightly less obvious first token. 💎

#### 2. Is it actually "Selection Bias"? 🧐
In the world of Recommender Systems, **Selection Bias** usually refers to a different problem: that our data only shows what people _already_ clicked on, not what they _would have_ clicked if they saw it.
Here we are biasing our "search" based on the model's first few guesses. This can be better described as **Inference Bias** or **Decoding Bias**. 

#### 3. Does Beam Search introduce Inductive Bias (preset assumptions)? 🐅
Yes, it does! While the **TIGER** model has its own biases (like assuming items with similar text are similar), the **Beam Search** adds a specific kind of "Search Bias."
The "bias" in Beam Search is that it assumes **locally good steps lead to a globally good result.** * It assumes that if the first half of a movie ID is very popular, the second half probably will be too.
- It "prefers" sequences that the model is confident about early on.

Beam Search is **pruning** (cutting off) paths. If the best recommendation starts with a "weird" or "rare" token, Beam Search might delete it before it gets a chance to shine.

However, researchers found something fascinating: **Beam Search has a "hidden" beneficial bias.** In language models, it has been shown to prefer text that has **Uniform Information Density**—meaning it generates results that feel more "human" and stable, rather than jumpy and chaotic.

#### 4. How TIGER and modern GRs fight this 🥊
To prevent the system from getting "stuck" in a boring, biased loop, researchers use a few tricks:
- **Diversity Constraints:** Some versions of beam search explicitly force the "beams" to be different. For example, "You can pick the top 3, but they must all come from different starting categories." 🌈
- **Temperature Sampling:** Before doing beam search, we can add "heat" (Temperature) to the model. This makes the probabilities flatter, giving lower-ranked tokens a better chance to make it into the top 3. 🌡️
- **Stochastic Beam Search:** Instead of just picking the top 3 "best," the model picks 3 tokens **randomly**, but gives the higher-probability ones a better "weight" or chance to be picked. This introduces a bit of "luck" and exploration! 🎲

