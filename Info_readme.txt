Ski jump

Game code license is 'Unlicense'. All library included or other files retain their licenses or rights. 

Game code includes files:

skijump_v2.exe (self extracting exe containing all files).

Files:

ski_competition.py
ski_geom.py
ski_jump.py
ski_physics.py
ski_scenery.py
ski_skier.py
ski_sound.py
test_competition.py
test_ski.py
test_skier.py
test_sound.py
ski_jumper_very_basic.obj
Wav sound formatted files.

ski_jump.exe is freeware, free to copy.

Most game code is written by Claude code, Sonnet 5.5 medium model.

Game start:
Open ski_jump.dist folder. Start ski_jump.exe .

Ski jumper 3d blender object external file to import (must reside in same folder to use): 
ski_jumper_very_basic.obj

---------------------------------
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org/>
----------------------------------

Dist build command:
python -m nuitka --standalone --follow-imports ski_jump.py

RELEASE folder:
ski_jump.dist

Ski jumper 3d blender object external file to import: 
ski_jumper_very_basic.obj