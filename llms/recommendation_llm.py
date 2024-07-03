
# pip install llama_index llama_index.embeddings.huggingface llama_index.llms.huggingface chromadb llama-index-vector-stores-chroma

import json
import chromadb
import time

from enums.offerings import *
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline



class RecommendationModel:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(RecommendationModel, cls).__new__(cls)
        return cls.instance

    def __init__ (self):
        self.TIME = time.time()

        self.EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
        self.EMBED_MODEL = HuggingFaceEmbedding(
            model_name=self.EMBED_MODEL_NAME,
            text_instruction="Given are the offers we provide. Emebed to capture the information about what each brand and offer provides",
            query_instruction="Retrieve all the relevent offers to the given query",
            cache_folder="./models/embedding_models/"+self.EMBED_MODEL_NAME,
        )


        self.CLASSIFIER_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
        self.CLASSIFIER = pipeline(
            task="zero-shot-classification",
            model=self.CLASSIFIER_NAME,
        )
        self.CLASSIFIER.save_pretrained(save_directory="./models/llm_models/"+self.CLASSIFIER_NAME)


        self.CHROMA_DB = chromadb.PersistentClient(path="./chroma")
        self.CHROMA_COLLECTION = self.CHROMA_DB.get_or_create_collection(
            name="Categories",
            metadata={"hnsw:space": "cosine"}
        )

        self.__notifyMessage("Model Initialized in "+str(time.time()-self.TIME)+"s")


    # Custom printer
    def __notifyMessage(self, text):
        print(f"\n{'='*20}{text}{'='*20} ")


    # Classifier
    def __generateResponseFromClassifier(self, userPrompt:str, threshold=0.51, use_tags=False):
        print("Using Tags: ", use_tags)
        self.TIME = time.time()


        return_ans = {
            "tags": [],
            "subtags": []
        }
        offer_subtags_array = []

        # If using tags, then the tags will be classified first
        if use_tags:
            # Classify the userPrompt into Tags, and add their subtags.
            tag_ans = self.CLASSIFIER(userPrompt, OFFER_TAGS_ARRAY, multi_label=True)
            for label, score in zip(tag_ans['labels'], tag_ans['scores']):
                if score >= threshold:
                    # Adding the tags to the final answer
                    return_ans['tags'].append(label)

                    # Adding the subtags associated with the current tag into a temp array
                    offer_subtags_array += list(OFFER_TAG_SUBTAG_DICT[label])


        # If no tags are identified, then take all the subtags
        if len(return_ans['tags']) == 0:
            offer_subtags_array = OFFER_SUBTAG_ARRAY


        # Classify the userPrompt into subtags
        subtag_ans = self.CLASSIFIER(userPrompt, offer_subtags_array, multi_label=True)
        for label, score in zip(subtag_ans['labels'], subtag_ans['scores']):
            if score >= threshold:
                # Adding the subtags to the final answer
                return_ans['subtags'].append(label)

        # If no label passed the threshold, then take the first 3
        if len(return_ans["subtags"])==0:
            return_ans["subtags"] = subtag_ans['labels'][:3]


        self.__notifyMessage("Classified in "+str(time.time()-self.TIME)+"s")

        # return_ans["subtags"].sort(descending=True)
        print("Classified Labels: ", return_ans)
        return return_ans


    # Will recreate all the embeddings and will store them in the ChromaDB
    def createSimpleChromaDB(self):
        file = open("./categories.json", "r")
        self.nodes = json.load(file)
        self.__notifyMessage("File Read")

        counter=0
        for i in self.nodes["data"]:
            counter+=1

            # The ID of the node
            curr_id = i["offering_id"]

            # The brand name of the node
            curr_brand_name = i["brand_name"].lower()
            if curr_brand_name[0].isspace():
                curr_brand_name = curr_brand_name[1:]
            if curr_brand_name[-1].isspace():
                curr_brand_name = curr_brand_name[:-1]


            # The metadata of each node, metadata is used for node filtering
            curr_metadata = {
                "offering_id": curr_id,
                "tag": "",
                "subtags": "",
                "brand_name": curr_brand_name,
            }

            # Filling the tags and subtags into the metadata
            for tag_i in i["tag"]:
                curr_metadata["tag"] = str(tag_i["name"])
                for sub_i in tag_i["subtags"]:
                    curr_metadata["subtags"] = str(sub_i["name"])


            # The actual node
            curr_node = {
                "details": i["details"].lower(),
                "brand_name": curr_brand_name,
                "tag": json.dumps(i["tag"])
            }
            if i["brand_intro"] != None:
                curr_node["brand_intro"] = i["brand_intro"].lower()


            # Converting the json node to string node
            json_str = json.dumps(curr_node)

            # Generating embeddings for the node
            curr_embeddings = self.EMBED_MODEL.get_text_embedding(json_str)

            # Adding to chroma collection
            self.CHROMA_COLLECTION.add(
                documents=[json.dumps(i)],
                embeddings=[curr_embeddings],
                metadatas=[curr_metadata],
                ids=[curr_id]
            )

            if counter%50 == 0:
                print("parsed ", counter, " nodes")

        # Saving to a zip file
        import shutil
        shutil.make_archive("chroma_db", 'zip', "./chroma")


    # Query the ChromaDB vector store
    def querySimpleChromaDB(self, userPrompt, top_res=5, use_tags=False):
        self.TIME = time.time()

        finalDataToReturn = []

        # user prompt preprocessing
        userPrompt = userPrompt.lower()
        print(userPrompt)

        # Embedding the user prompt
        embedded_text = self.EMBED_MODEL.get_query_embedding(userPrompt)

        # finding if brand name is present
        brandsFoundEmbeddings = []
        brandsFoundNames = []
        for word in userPrompt.split(" "):
            if word in BRAND_NAMES_SET:
                brandsFoundEmbeddings.append(self.EMBED_MODEL.get_query_embedding(word))
                brandsFoundNames.append(word)

        print(brandsFoundNames)

        # If brand name is found, query it.
        if len(brandsFoundEmbeddings) > 0:
            ans = self.CHROMA_COLLECTION.query(
                query_embeddings=brandsFoundEmbeddings,
                n_results=10,
                include=["metadatas", "documents"],
                where={"brand_name": {"$in": brandsFoundNames}}
            )
            for i in ans["documents"][0]:
                finalDataToReturn.append(json.loads(i))

            print(ans)
        # return {"documents": finalDataToReturn}

        # Classifying the user prompt into tags and subtags.
        classified_labels_dict = self.__generateResponseFromClassifier(userPrompt, use_tags=use_tags)

        # The clause to use for chroma DB filtering
        clause = {"$or": [

        ]}

        if use_tags:
            print("Using tags")
            temp_clause = {}
            if classified_labels_dict["tags"] != []:
                clause["$or"].append({"tags": {"$in": classified_labels_dict["tags"]}})
                temp_clause = {"tags": {"$in": classified_labels_dict["tags"]}}
            if classified_labels_dict["subtags"] != []:
                clause["$or"].append({"subtags": {"$in": classified_labels_dict["subtags"]}})
                temp_clause = {"subtags": {"$in": classified_labels_dict["subtags"]}}
            if len(clause["$or"])!=2:
                clause = temp_clause

        else:
            print("Using subtags")
            clause = {"subtags": {"$in": classified_labels_dict["subtags"]}}


        print("clause:  ", clause)

        ans = self.CHROMA_COLLECTION.query(
            query_embeddings=[embedded_text],
            n_results=top_res,
            include=["metadatas", "documents"],
            where=clause
        )

        self.__notifyMessage(f"Query Time: {time.time()-self.TIME}s")

        for i in ans["documents"][0]:
            finalDataToReturn.append(json.loads(i))

        return {"documents": finalDataToReturn}


if __name__ == "__main__":
    print("Starting Locally")
    
    l = RecommendationModel()

    while(True):
        input_prompt = input("What are you looking for (Empty for default prompt)? ")
        if input_prompt == "":
            input_prompt = "I am going to a party and I want to surprise my friend with a sweet dish I prepared for him."
        if(input_prompt == "exit"):
            break
        
        print("Prompt: ", input_prompt)
        docs_array = l.querySimpleChromaDB(input_prompt, 5, use_tags=True)
        print(docs_array)



