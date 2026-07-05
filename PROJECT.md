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
on agent but not vice versa. The genealogy directory treats agent like a library.

