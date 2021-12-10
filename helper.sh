#!/bin/bash

# =============================================================================
# HELPER ACTIONS
# =============================================================================

NC=$(echo "\033[m")
BOLD=$(echo "\033[1;39m")
CMD=$(echo "\033[1;34m")
OPT=$(echo "\033[0;34m")

action_usage(){
    echo -e "  ____                _____                    "
    echo -e " |  _ \  ___  ___ _ _|_   _| __ __ _  ___ ___  "
    echo -e " | | | |/ _ \/ _ \ '_ \| || '__/ _\` |/ __/ _ \ "
    echo -e " | |_| |  __/  __/ |_) | || | | (_| | (_|  __/ "
    echo -e " |____/ \___|\___| .__/|_||_|  \__,_|\___\___| "
    echo -e "                 |_|                           "
    echo -e ""                                          
    echo -e "${BOLD}System Commands:${NC}"
    echo -e "   ${CMD}init${NC} initializers environment;"
    echo -e "   ${CMD}test${OPT} ...${NC} runs tests;"
    echo -e "      ${OPT}-m <MARK> ${NC}runs tests for mark;"
    echo -e "      ${OPT}-c ${NC}generates code coverage summary;"
    echo -e "      ${OPT}-r ${NC}generates code coverage report;"
    echo -e "   ${CMD}generate${OPT} -h${NC} generates synthetic dataset;"
    echo -e "   ${CMD}transform${OPT} -h${NC} transforms source to TFRecords;"
    echo -e "   ${CMD}docs${NC} generates documentation;"
    echo -e "   ${CMD}build${NC} generates distribution archives;"
    echo -e "   ${CMD}stage${NC} deploy DeepTrace to Test Python Package Index;"  
}

action_init(){
    # if [ -d .venv ];
    #     then
    #         rm -r .venv
    # fi

    # python3 -m venv .venv
    source .venv/bin/activate 


    # if [[ -f dependencies.txt ]]
    # then
    #     pip3 install -r dependencies.txt --no-cache
    # else
    #     pip3 install -r requirements.txt --no-cache
    # fi
    # pip3 install rich
    # pip3 install -i https://test.pypi.org/simple/ hungarian-loss
    # pip3 install ../hungarian-loss/dist/hungarian_loss-1.0.1.dev0+g942dbfe.d20211122-py3-none-any.whl
    # pip3 install tensorflow-addons==0.14.0
    # pip install ../hungarian-loss/dist/hungarian_loss-0.1.dev30+g61d4093.d20211204-py3-none-any.whl
    pip install hungarian-loss
}

action_test(){
    source .venv/bin/activate

    OPTS=()
    while getopts ":m:cr" opt; do
        case $opt in
            m)
                OPTS+=(-m $OPTARG) 
                ;;
            c)
                OPTS+=(--cov=deeptrace) 
                ;;
            r)
                OPTS+=(--cov-report=xml:cov.xml) 
                ;;
            \?)
                echo -e "Invalid option: -$OPTARG"
                exit
                ;;
        esac
    done
    
    pytest --capture=no -p no:warnings ${OPTS[@]}
}

action_dataset(){
    source .venv/bin/activate
    python main.py dataset ${@}
} 

action_tfrecords(){
    source .venv/bin/activate
    python main.py tfrecords ${@}
}

action_train(){
    source .venv/bin/activate
    python main.py train ${@}
}

action_predict(){
    source .venv/bin/activate
    python main.py predict ${@}
}

action_install(){
    source .venv/bin/activate
    pip install ${@}
}

action_mkdocs(){
    source .venv/bin/activate
    mkdocs serve
}

action_build(){
    source .venv/bin/activate
    python -m build
}

action_stage(){
    source .venv/bin/activate
    read -p "Do you wish to stage DeepTrace to https://test.pypi.org (y/n)? " answer
    case ${answer:0:1} in
        y|Y )
            python3 -m twine upload --repository testpypi dist/*
        ;;
        * )
            echo -e "Aborted!"
        ;;
    esac
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    init)
        action_init
    ;;
    test)
        action_test ${@:2}
    ;;
    dataset)
        action_dataset ${@:2}
    ;;
    tfrecords)
        action_tfrecords ${@:2}
    ;;
    train)
        action_train ${@:2}
    ;;
    predict)
        action_predict ${@:2}
    ;;
    install)
        action_install ${@:2}
    ;;
    mkdocs)
        action_mkdocs ${@:2}
    ;;
    *)
        action_usage
    ;;
esac  

exit 0