# deep-pitcher
utility for pitch estimation by vocal separation from audio source with machine learn

## Initialization
### .itdb.config 설정
디렉토리 내의 **itdb.config.example** 파일을 수정한 후 파일명을 **.itdb.config**로 변경합니다.
config 파일 내에서는 불러오고자 하는 iTunes XML 파일 위치와 MySQL 연결 정보를 입력해야 합니다.  
MySQL 연결 정보에서 username, password, database name의 기본값은 모두 itdb입니다.
### MySQL에서 새로운 유저 만들기 및 권한 부여
SQL 서버에 root계정으로 로그인 해 다음 쿼리(기본값 itdb 기준)를 실행합니다.
~~~ sql
CREATE USER 'itdb' IDENTIFIED BY 'itdb';
GRANT ALL PRIVILEGES ON itdb.* TO 'itdb' WITH GRANT OPTION;
GRANT FILE on *.* to itdb;
FLUSH PRIVILEGES;
~~~
## How to use
1. Initialization에 있는 과정을 수행합니다.
2. import_file.py를 실행하여 iTunes xml의 내용을 MySQL에 저장합니다.
3. To be Continued...
## Error 해결법
### ERROR CODE 3948 (Loading local data disabled) 해결법
~~~  sql
Mysql -u root -p
Use mysql;
SELECT @@local_infile;
; Local_infile 값이 0이면 1로 수정해야 함
SET @@GLOBAL.local_infile:= 1;
~~~
### OperationalError(1290) secure_file_priv 해결법
MySQL 상에서 다음을 실행하여 값을 확인합니다.
~~~ sql
SELECT @@GLOBAL.secure_file_priv;
~~~
* NULL: 어떠한 경로도 지정돼 있지 않다; 어떤 파일이든 읽고 쓸 수 없다.
* 특정 경로: 해당 경로의 파일은 읽고 쓸 수 있다.
* 아무 값이 표시되지 않음(empty): 어떤 경로의 파일이든 읽고 쓸 수 있다.  
참조: https://dev.mysql.com/doc/refman/5.7/en/secure-file-priv.html

값이 NULL이거나 특정 경로가 있을 경우에는 해당 값을 수정해야 합니다. 이를 위해서는 .cnf나 .ini파일에 다음과 같은 내용을 추가해야 합니다.  
~~~ bash
[mysqld]
secure_file_priv="" or "/tmp" 
~~~
deep-pitcher에서 기본적으로 SQL로 로드하는 파일은 /tmp에 위치합니다. 따라서 "/tmp"를 입력하는 것이 (그나마) 보안상 좋으나, SSH 등의 원격 서버에 연결했을 경우는 ""(empty)를 입력해야 합니다.
* macOS에서 dmg로 mysql을 설치한 경우:
    * 위의 내용을 담은 새로운 cnf파일을 생성합니다.
    * 설정→MySQL→Configuration→Configuration File에 체크 후 cnf파일 경로 입력
* linux 및 macOS(brew 등) 환경:
    * vim /etc/mysql/my.cnf
    * 위의 내용 추가한 후 저장
    * MySQL 서버 재시작 (rpm으로 설치한 경우 systemctl restart mysqld)
  
SELECT @@GLOBAL.secure_file_priv; 시 원하는 경로로 값이 바뀌어 있으면 성공입니다.