### DL repos

- for cloning as submodule: `git submodule add <submodule>`
- separate conda env used for each of the frameworks, saved in registrationbaselines/<submodule>_requirements.txt
- -for cloning this repo together with submodules: `git clone --recurse-submodules <submodule>`
    - if it is alread cloned 'normally' then use
        `git submodule init && git submodule update` to pull the submodules