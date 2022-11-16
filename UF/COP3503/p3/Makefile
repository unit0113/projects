OBJS = src/*.cpp
INCLUDE_PATH = -I"C:\SFML-2.5.1\include"
LIBRARY_PATH = -L"C:\SFML-2.5.1\lib"
LINKER_FLAGS = -DSFML_STATIC -static-libgcc -static-libstdc++ -lmingw32 -lsfml-graphics-s -lsfml-window-s -lsfml-audio-s -lsfml-system-s -lopengl32 -lfreetype -ljpeg -lwinmm -lgdi32 -lopenal32 -lFLAC -lvorbisenc -lvorbisfile -lvorbis -logg


main: $(OBJS)
	g++ -std=c++17 $(OBJS) $(INCLUDE_PATHS) $(LIBRARY_PATH) $(LINKER_FLAGS) -o project3.exe

clean:
	del project3.exe