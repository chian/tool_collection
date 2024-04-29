import json
import uuid
import time
from functools import wraps
import concurrent.futures
from langchain_core.prompts import ChatPromptTemplate

DEBUG = 1
ID_LIMIT = 5

# FUNCTION DEFINITIONS
def call_with_timeout(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=120)  # 30 seconds timeout
        except concurrent.futures.TimeoutError:
            print("The API call timed out.")
            raise Exception("API call timed out.")
        
def retry_on_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                result = func(*args, **kwargs)
                if result is None:
                    print("No result returned. Retrying the API call in 10 minutes...")
                    # Save checkpoint before sleeping
                #    checkpoint_data = {
                #        "agenticChunker": ac.to_json(),
                #        "currentFileIndex": current_file_index,
                #        "currentParagraphIndex": current_paragraph_index,
                #        # Include other relevant data in the checkpoint
                #    }
                #    save_checkpoint(checkpoint_data, "checkpoint.json")
                    time.sleep(360)  # Sleep for 1 hour before retrying
                else:
                    return result
            except Exception as e:
                print(f"An error occurred: {e}. Retrying the API call in 10 minutes...")
                # Save checkpoint before sleeping
                #checkpoint_data = {
                #    "agenticChunker": ac.to_json(),
                #    "currentFileIndex": current_file_index,
                #    "currentParagraphIndex": current_paragraph_index,
                    # Include other relevant data in the checkpoint
                #}
                #save_checkpoint(checkpoint_data, "checkpoint.json")
                time.sleep(360)  # Sleep for 1 hour before retrying
    return wrapper

@retry_on_error
def robust_api_call(func, *args, **kwargs):
    while True:
        try:
            result = call_with_timeout(func, *args, **kwargs)
            if result is None:
                raise Exception("API call failed or timed out.")
            return result
        except ConnectionError as e:
            print(f"ConnectionError: {e}")
            print("Service cannot be found. Pausing for 600 seconds before retrying...")
            # Save checkpoint before sleeping
            checkpoint_data = {
                "agenticChunker": ac.to_json(),
                "currentFileIndex": current_file_index,
                "currentParagraphIndex": current_paragraph_index,
                # Include other relevant data in the checkpoint
            }
            save_checkpoint(checkpoint_data, "checkpoint.json")
            time.sleep(600)

class AgenticChunker:
    def __init__(self, llm, chunk_summary_instruction=None):
        self.chunks = {}
        self.id_truncate_limit = ID_LIMIT
        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = DEBUG
        self.llm = llm
        self.llm_call_count = 0
        self.chunk_summary_instruction = chunk_summary_instruction

    def to_json(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_data, llm=None, chunk_summary_instruction=None):
        data = json.loads(json_data)
        ac = cls(llm)
        ac.chunks = data["chunks"]
        ac.id_truncate_limit = data["id_truncate_limit"]
        ac.generate_new_metadata_ind = data["generate_new_metadata_ind"]
        ac.print_logging = data["print_logging"]
        ac.llm_call_count = data["llm_call_count"]
        ac.chunk_summary_instruction = chunk_summary_instruction
        return ac

    def add_propositions(self, propositions, file_path):
        for proposition in propositions:
            self.add_proposition(proposition, file_path)

    def add_proposition(self, proposition,file_path):
        if self.print_logging:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print ("No chunks, creating a new one")
            self._create_new_chunk(proposition,file_path)
            return

        chunk_id = self._find_relevant_chunk(proposition)
        #chunk_id = robust_api_call(self._find_relevant_chunk, proposition)
        if DEBUG:
            print("chunk_id (add_proposition):", chunk_id)

        # If a chunk was found then add the proposition to it
        if chunk_id is not None:
            if "create" in chunk_id.lower():
                if self.print_logging:
                    print("Creating a new chunk as suggested by the LLM")
                self._create_new_chunk(proposition, file_path)
            else:
                if self.print_logging:
                    print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
                self.add_proposition_to_chunk(chunk_id, proposition)
                #robust_api_call(self.add_proposition_to_chunk, chunk_id, proposition)
        else:
            if self.print_logging:
                print ("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition,file_path)


    def add_proposition_to_chunk(self, chunk_id, proposition):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = robust_api_call(self._update_chunk_summary, self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = robust_api_call(self._update_chunk_title, self.chunks[chunk_id])
            #self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            #self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = robust_api_call(runnable.invoke, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        })
        self.llm_call_count += 1
        """
        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        }).content
        """
        return new_chunk_summary

    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = robust_api_call(runnable.invoke, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        })
        self.llm_call_count += 1
        """
        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        }).content
        """
        if DEBUG:
            print("updated_chunk_title:",updated_chunk_title)
        return updated_chunk_title


    def _create_new_chunk(self, proposition,file_path):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] # I don't want long ids
        #new_chunk_id = "aAa" + new_chunk_id + "aAa" #add delimiters to make it easier to find
        if DEBUG:
            print("new_chunk_id (_create_new_chunk)",new_chunk_id)
        new_chunk_summary = robust_api_call(self._get_new_chunk_summary, proposition)
        new_chunk_title = robust_api_call(self._get_new_chunk_title, new_chunk_summary)
        #new_chunk_summary = self._get_new_chunk_summary(proposition)
        #new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks),
            'file_path' : file_path,
        }
        if self.print_logging:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def _get_new_chunk_summary(self, proposition):
        if self.chunk_summary_instruction==None:
            specific_instructions="""    
            Your summaries should anticipate generalization. 
            If you get a proposition about apples, generalize it to food.
            Or month, generalize it to "date and times".        
        
            Example:
            Input: Proposition: Greg likes to eat pizza
            Output: This chunk contains information about the types of food Greg likes to eat.
            """
        else:
            specific_instructions = self.chunk_summary_instruction

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    {specific_instructions}
                    
                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = robust_api_call(runnable.invoke, {
            "proposition": proposition,
            "specific_instructions" : specific_instructions
        })
        self.llm_call_count += 1
        """
        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        })
        """
        return new_chunk_summary

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = robust_api_call(runnable.invoke, {
            "summary": summary
        })
        self.llm_call_count += 1
        """
        new_chunk_title = runnable.invoke({
            "summary": summary
        })
        """
        if DEBUG:
            print("new_chunk_title:",new_chunk_title)
        return new_chunk_title

    def get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\nFile: {chunk['file_path']}\n\n"""

            chunk_outline += single_chunk_string

        return chunk_outline


    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.
                    
                    A proposition should belong to a chunk of their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.
                    
                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"
                    
                    Example:
                    Input:
                    - Proposition: "Greg really likes hamburgers"
                    - Current Chunks:
                    - Chunk ID: 2n4l3d
                    - Chunk Name: Places in San Francisco
                    - Chunk Summary: Overview of the things to do with San Francisco Places
                    
                    - Chunk ID: 93833k
                    - Chunk Name: Food Greg likes
                    - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = robust_api_call(runnable.invoke, {
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        })
        self.llm_call_count += 1

        if DEBUG:
            print("chunk_found (_find_relevant_chunks):",chunk_found)

        existing_chunk_ids = list(self.chunks.keys())
        for existing_id in existing_chunk_ids:
            if existing_id in chunk_found:
                return existing_id

        return None

    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")

        # Group propositions by chunk ID
        grouped_propositions = {}
        for chunk_id, chunk in self.chunks.items():
            for prop in chunk['propositions']:
                if chunk_id not in grouped_propositions:
                    grouped_propositions[chunk_id] = []
                grouped_propositions[chunk_id].append(prop)

        # Print propositions grouped by chunk ID
        for chunk_id, propositions in grouped_propositions.items():
            chunk = self.chunks[chunk_id]
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in propositions:
                print(f"    - {prop}")
            print("\n")

    def pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self.get_chunk_outline())