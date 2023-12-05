#[[
    @brief  Download dependency library from git repository.
    @param  url - git repository address
    @param  branch - the branch or tag name
    @param  lib_name - the expected local name
]]
function(pipe_download_dependency url branch lib_name work_directory)
    execute_process(COMMAND bash "-c" "if [ ! -d ${work_directory} ];\
        then mkdir -p ${work_directory}; fi")

    execute_process(COMMAND bash "-c" "if [ ! -d ${lib_name} ]; then \
        git clone --progress -q --depth=1 -b ${branch} ${url} ${lib_name}; fi"
        WORKING_DIRECTORY ${work_directory})
endfunction()

MACRO(SUBDIRLIST RESULT CURDIR SEARCH_FLAG)
SET( _CURDIR ${CURDIR} ${ARGN} ) 
    LIST(APPEND ${RESULT} PUBLIC ${_CURDIR})
    FOREACH(SUBDIR ${_CURDIR})

        IF(${SEARCH_FLAG} STREQUAL "0")#no search 
            SET(CHILDREN )
        ELSEIF(${SEARCH_FLAG} STREQUAL "1")#search sub direcory 
            FILE(GLOB CHILDREN RELATIVE ${SUBDIR} ${SUBDIR}/*)
        ELSEIF(${SEARCH_FLAG} STREQUAL "2") #search all 
            FILE(GLOB_RECURSE CHILDREN LIST_DIRECTORIES true RELATIVE ${SUBDIR} ${SUBDIR}/*)
        ENDIF()

        FOREACH(CHILD ${CHILDREN})
            IF(IS_DIRECTORY ${SUBDIR}/${CHILD})
                ##MESSAGE(${SUBDIR}/${CHILD})
                LIST(APPEND ${RESULT} PUBLIC ${SUBDIR}/${CHILD})
            ENDIF()
        ENDFOREACH()

    ENDFOREACH()
ENDMACRO()