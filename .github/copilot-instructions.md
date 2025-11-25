1.  Names
    Names should give useful information about the variable or function.
1.1 Variables
a.  Use meaningful names. 
b.  Important API level and global variables shall be prefixed with aoclfftz_ .
c.  Use underscores (snake case) to separate words in a name.
d.  Stick to lower case.
e.  Local variable names can be shorter and their purpose can be explained with comments.
f.  place the * close to the variable name not pointer type.
g.  Include units in variable names, eg : timeout_msecs
h.  Pointers shall be prefixed with p_ & function pointers shall be suffixed with _fp
i.  Use const and static keyword everywhere applicable

1.2 Functions
a.  Prefix all the interface APIs with aoclfftz_ .
b.  Sometimes, concrete module level interface APIs can be prefixed with modulename_ .
c.  Make function names as verbs that perform an action to distinguish them from data objects. For Example: Use check_for_errors() instead of error_check()
d.  Function parameter order : inputs, then outputs.

1.3 Macros and Global constants
a.  Use upper case (screaming snake case) with “_” separators.
b.  Important API level and global variables shall be prefixed with AOCLFFTZ_.
c.  Wrap the macro expression in parenthesis to avoid wrong order of operations.
d.  In case a macro definition is quite big, consider using an inline function provided there are no performance side-effects. For multi-line macros, prefer do-while(0). If do-while(0) is not used, and curly braces are used, the auto-format will leave 1 tab space for the macro body.

1.4 Enums
a.  Use “_” separators for all the labels.
b.  Important API level and global enums shall be prefixed with aoclfftz_ .

1.5 Structures
a.  Use lower case with “_” separators for the structure names.
b.  Important API level structures shall be prefixed with aoclfftz_ .
c.  Organize the member variables in the structure in the decreasing order of their sizes to minimize memory alignment issues.

2.  Formatting 
a.  Maintain 80 chars a line and use a tab size of 4 spaces.
b.  Use the braces form where the open brace starts in the new line.
c.  Always enclose if or loop constructs with curly braces. Add comments to the closing braces.
d.  Variable declarations shall be followed by a new line to separate them from the rest of the code.
e.  No space should be added after the type and the variable when typecasting it.
f.  Always initialize the variables.
g.  Use header file guards to avoid multiple inclusions.
h.  One statement per line.
i.  One variable defined per line.
j.  Try to avoid goto statements. But, when it is really necessary, write good comments and place the label to the left of the code that follows.
k.  Allman style shall be used for all Bracing and indentation.
l.  Leave a space between if/while & (
m.  System headers followed by project specific headers.
	
3.  Comments
a.  Write a brief comment (in doxygen style) at the start of each source file about the overall purpose of the file.
b.  Write comments (in doxygen style) for each function providing details about what the function does, what are its arguments along with the possible values/range of arguments and their meanings, and what are the return values/codes of this function.
c.  Write comments (in doxygen style) for macros and enums.
d.  Write comments (in doxygen style) for important variables including the global variable.
e.  Write comments (in doxygen style) for important structures.
f.  Write comments for every #endif
g.  Write elaborate comments for function or code section in a function that needs explanation about the algorithmic steps or strategies adopted.

4.  Avoid
a.  `goto` statements (use only if absolutely necessary with clear comments)
b.  Magic numbers (use named constants)
c.  Deep nesting (refactor into helper functions)
d.  Long functions (aim for < 200 lines)
