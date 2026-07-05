Introduction
----------------
This project is an experiment in agentic AI. The goal is to discover a useful and flexible 
approach that can be used for multiple projects and multiple types of agentic programming.

A specific use case was added, but it is loosely coupled to the core agentic code, which is 
meant for eventual broader use. This is performing queries in genealogy. This is discussed more 
below.

One choice was to avoid currently popular frameworks and libraries such as LangChain and LangGraph.
The point of this was multifold.

One reason was that they products take away a lot of flexibility and coax you into a particular 
pattern which can be limiting. They also perform so many actions for you that you don't understand 
what is really happening. This might limit your ability to control and explain what is happening. 
It might prevent you from truly fixing problems that arise. It also simply reduces the
learning that occurs. Finally, it's likely that we are too early into agentic AI and such 
frameworks are unlikely to remain popular for long.

For these reasons, I decided to build this myself using minimal dependencies. This kind of application
is often called a harness or agentic program.

Another main goal is not to discover a universal approach to all agentic business problems but to discover some
core approaches that will be useful for particular ones I might face; problems that are challenging and, to me, 
more interesting, such as some of the problems encountered in robotics.

Dependencies
------------------

The main dependency is the Anthropic Python package. This brings you the basic SDK that handles 
the following concerns.

* Enables the use of the frontier LLMs (obviously not building those myself)
* Wraps the LLM models in a class that makes http requests to the anthropic models on their cloud
* Uses the Anthropic tool chaining protocol which allows you to inform the model of local tools 
  that are available, allows it to suggest calls to those in a way that can be captured and run before 
  handing the results back to the LLM for further processing

The tool chaining mechanism allows for construction of loops. The looping and control over it happens
on the application side, tis harness. The model side where the LLM lives contains no statefulness. 
It also does not actually run tools. The agentic harness (my application code) does that. LLMs only 
output tokens. The harness interprets those however it wishes. It chooses whether to run tools. I can 
apply whatever checks, constraints, guardrails and security it needs. Being able to do this explicitly 
is another reason to avoid large frameworks.

There a few other minor dependencies added. Pydantic is used in a few places but is also already a 
dependency of anthropic packages. The Python package disckcache is used to cache model calls locally.
This uses SQLite internally is basically a simple key-value store made specifically for caching. SQLite
is also used for the genealogy use case. The ged4py is a python package for reading GEDCOM files which
is a file format for genealogical data.

First use case: Genealogy
---------------------------

It doesn't make sense developing an agentic framework without at least a good use case, so I chose the
following based on a hobby of mine: family genealogy. Geni.com is one of many genealogy websites where 
you can download your family tree, if it exists. Such trees are build by both professional and 
amateur genealogists. Sites like geni.com or ancestry.com are useful for browsing the tree or the data 
on the web. But I wanted to do programmatic searches on the raw data. Furthermore, I wanted to use the
power of LLMs to pick up where rules based code falls short or becomes tedious.

The data can be exported into GEDCOM files, a text based exchange format that is basically a list of 
facts and relations arranged into hierarchies. It looks like this:

0 @I75@ INDI
1 NAME Waldemar  //
1 SEX M
1 BIRT
2 DATE        1868
1 DEAT
2 DATE        1879
1 FAMC @F3@
0 @I76@ INDI
1 NAME Sophie of_Prussia //
1 TITL Queen of Greece
1 SEX F

I have two GEDCOM files for my family (all blood relatives, limited in number by Geni) and director ancestors
(e.g. all great-great-grandparents but no uncles, cousins, siblings etc.). I also have a few public
GEDCOMs (English royal family, William Shakespeare Family, Bronte family, US Presidents etc.)

These GEDCOMs needs to be parsed and turned into some useful data structures that can be queried more easily. 

Storage:

I chose to use ged4py to read the GEDCOM files and then create some Pydantic objects in 3rd normal form; 
and then load these into SQlite. A relational database has plenty of obvious advantages. SQLite is a 
good choice since the data is small, write-once/read-many, is preinstalled most places, there is not 
yet any need to support multiple users on the cloud etc.

SQL is also a good language for LLM tool use as LLMs are excellent at writing SQL, especially on fairly simple
data models. 

There are currently 5 tables:

* individuals
* families
* family_children
* individual_events
* family_events

The while schema with indices is currently only 45 lines of SQL code. It may grow as we add other information
from the GEDCOM.

When given a well-written system prompt including the tool defined around SQL and the database schema, 
we build a genealogical agent around this core agent code which can answer questions like:

"How many uncles does David Johnston have?"

or (for the royal dataset)

"Has there ever been a King of England that was born in Denmark? If so, did they have any children
that became King or Queen?"

There are other types of queries that may require other tools beyond SQL search such as those better
suited to walking a tree graph.

Design Characteristics
-------------------------

The package gene has two subdirectories agent and genealogy. These are not yet separated into
uv packages (or separate repos etc) but are roughly considered separate with genealogy depending 
on agent but not vice versa. The genealogy directory treats agent like a library. This will likely 
evolve in the future but keep loose coupling now is important for that.

There are several high level goals for the design:

* Simplicity - Agentic AI applied to use cases can be complex enough on it's own. Don't add extra complexity
* Understandability - You need to understand what your code is doing always
* Flexibility - As you address harder parts of the problem or other hard problems, you will need to take 
  some different approaches but want to try to reuse a number of flexible components in different ways

These drive some lower level needs:

* Observability - You need to make it easy to see what happened. This needs to drive the design not be added on later.
* Testability - Unit test. Integration tests. Ability to run eval suites and see the effect of any code or 
  configuration changes
* Reign in or control statefulness - Statefulness is not entirely avoidable but you can limit it to a few obvious 
  places and keep most of the code functional (referentially transparent). This aids understandability, and reduces 
  complexity and makes testing easier.

These features and needs drove decisions such as the following:
* Cache the LLM model calls to reduce non-determinism and handle the cache carefully
* Make a "Turn" object, a fundamental building block, containing multiple LLM trips (Steps which 
  can include tool calls) that is stateless (when LLMs are cached).
* Push the state required for carrying on longer context-bound conversations and other modes of complex thinking 
  to higher layers that use stateless Turns.

Future additions
------------------

The current design makes it fairly straight forward to consider new features if they are justified by 
used cases.

Complex handling of memory - The Turn object is stateless. The conversation loop is just one simple way 
of extending it and adding not only a loop but the stateful concept. For example right now, the Conversation
class calls a Turn with a query, and runs the Turn which can contain tool calls within and multiple trips to
the LLM. The Conversation then decides what to put into history so that the LLM is aware of certain context.
This is the right seam to introduce the more complex idea of Memory. 

Currently it just consists of taking the messages out of the turn and accumulating those and handing them 
back to the model. That is simple but might not scale to very long conversations. The LLM context will 
eventually become pressured or actually fill and performance will drop.

To handle that, we can do things like introduce a more complex Memory handler that compacts the context. 
Examples might include summaries of older messages, dropping older ones. Might also include creating 
high level summaries like a table of contents and creating an explicit tool call which effectively 
allows the model to gain details from older memory. Thus, memory can become a kind of database and the LLM
can choose to make use of it if needed. 

We don't want to add this unless a use case indicates it is needed but we have a flexible design that 
should be able to support it. 

Layered modes of thought - We can add additional layers that make use of Turns in more complex ways.
This includes having multiple agents with different systems prompts which can be directed to act together.
Keeping this well structured and understandable is likely a challenge more so that simply making the connections.

You can also have a single agent with categories of thought modeled as a state machine. They can be layered
so that they have a natural ordering, e.g. high-level planning is above execution and you move up and down the 
stack or unordered: creative exploration, verifying results, fixing mistakes etc.

All of these models can make use of the same stateless Turn tool and just build on their own concerns. 
This is an example of separation of concerns and a way to encapsulate the complexity of everything we 
have built so far so that it does not interact strongly with whatever new complexity needs to come next.

Other design choices
----------------------
This currently supports only Anthropic models. Adding support for OpenAI models, local models etc. is possible
but I delayed this for a few reasons. The main one is that is adds some complexity or the need for adpators or
possibly a unifying library like LiteLLM and it doesn't really help us on our goals. Anthropic models are very
competitive and still offer trade off choices between accuracy and speed/cost. In have built such adapters in 
the past and it doesn't really change much. Need for other models might come later and would be half a day's work
at the cost of some added complexity to the core.

It is CLI driven only. Makes little sense to add things like visualization or UIs until the logic has stabilized
and patters become more ingrained.

TODO items
-------------

No need for Docker or containers yet. This is still experimental and intended to be run locally. Still has a 
reliable build.

The data is not yet attached anywhere to the repo. It is manually installed into a git-ignored directory.
Moving it to it's own repo or somewhere it can be loaded from is a TODO item. It's all public data anyways.
The genealogy evals cannot be run until the data is installed.

Other Design details
---------------------

Test running:

Loading genealogy data:

Running evals:

Observability:

Model config:

API key config:

Security/safety issues:

Other:

