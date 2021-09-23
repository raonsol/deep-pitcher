-- Created by: Alex K (wtwf.com) Sun Mar  5 10:32:41 2006
-- Copyright 2006 all rights reserved
-- here's the schema for a databse to hold itunes library data
-- I've added a column called User_ID to allow the database to store
-- more than one person's library
SET SQL_MODE = 'ALLOW_INVALID_DATES';
-- right now we only store track info and playlist info
DROP TABLE IF EXISTS tracks;
CREATE TABLE tracks (
    User_ID INTEGER(4) NOT NULL,
    Track_ID INTEGER(4) NOT NULL,
    Name VARCHAR(1024) DEFAULT '',
    Artist VARCHAR(256) DEFAULT '',
    Comments VARCHAR(1024) DEFAULT '',
    Album VARCHAR(256) DEFAULT '',
    Album_Artist VARCHAR(256) DEFAULT '',
    Composer VARCHAR(256) DEFAULT '',
    Sort_Series VARCHAR(256) DEFAULT '',
    Sort_Name VARCHAR(256) DEFAULT '',
    Sort_Album VARCHAR(256) DEFAULT '',
    Sort_Artist VARCHAR(256) DEFAULT '',
    Series VARCHAR(256) DEFAULT '',
    Episode_Order INTEGER(4),
    Episode VARCHAR(256) DEFAULT '',
    Genre VARCHAR(64) DEFAULT '',
    Content_Rating VARCHAR(128) DEFAULT '',
    Kind VARCHAR(32) DEFAULT '',
    Size BIGINT(8) UNSIGNED,
    Play_Date BIGINT(8) UNSIGNED,
    Total_Time INTEGER(4),
    BPM INTEGER(4),
    Disc_Number INTEGER(1),
    Disc_Count INTEGER(1),
    Track_Number INTEGER(2),
    Track_Count INTEGER(2),
    Year INTEGER(2),
    Compilation BOOLEAN default false,
    Has_Video BOOLEAN default false,
    Purchased BOOLEAN default false,
    Protected BOOLEAN default false,
    TV_Show BOOLEAN default false,
    Movie BOOLEAN default false,
    Disabled BOOLEAN default false,
    Album_Rating_Computed BOOLEAN default false,
    Rating_Computed BOOLEAN default false,
    Part_Of_Gapless_Album BOOLEAN default false,
    Loved BOOLEAN default false,
    Music_Video BOOLEAN default false,
    Podcast BOOLEAN default false,
    Unplayed BOOLEAN default false,
    Explicit BOOLEAN default false,
    Clean BOOLEAN default false,
    Start_Time INTEGER(4) UNSIGNED,
    Stop_Time INTEGER(4) UNSIGNED,
    Volume_Adjustment INTEGER(1),
    Sort_Album_Artist VARCHAR(256),
    Sort_Composer VARCHAR(256),
    Date_Modified DATETIME NOT NULL DEFAULT '0000-00-00',
    Release_Date DATETIME NOT NULL DEFAULT '0000-00-00',
    Date_Added datetime NOT NULL DEFAULT '0000-00-00',
    Skip_Date datetime NOT NULL DEFAULT '0000-00-00',
    Skip_Count INTEGER(1) DEFAULT 0,
    Bit_Rate INTEGER(2),
    Sample_Rate INTEGER(2),
    Play_Count INTEGER(2),
    Play_Date_UTC datetime NOT NULL DEFAULT '0000-00-00',
    Album_Rating INTEGER(1) DEFAULT 0,
    Rating INTEGER(1) DEFAULT 0,
    Artwork_Count INTEGER(1) DEFAULT 0,
    Season INTEGER(1),
    Persistent_ID VARCHAR(20),
    Track_Type VARCHAR(8),
    File_Type INTEGER(4),
    File_Creator INTEGER(4),
    Location VARCHAR(512),
    File_Folder_Count INTEGER(1),
    Library_Folder_Count INTEGER(1),
    PRIMARY KEY (User_ID, Track_ID)
);
DROP TABLE IF EXISTS playlists;
CREATE TABLE playlists (
    User_ID INTEGER(4) NOT NULL,
    Playlist_ID INTEGER(4) NOT NULL,
    Name VARCHAR(1024),
    Playlist_Persistent_ID VARCHAR(1024),
    Parent_Persistent_ID VARCHAR(1024),
    -- Folder  boolean - how do we do this?
    PRIMARY KEY (User_ID, Playlist_ID)
);
DROP TABLE IF EXISTS playlist_tracks;
CREATE TABLE playlist_tracks (
    User_ID INTEGER(4) NOT NULL,
    Playlist_ID INTEGER(4) NOT NULL,
    Track_ID INTEGER(4) NOT NULL,
    PRIMARY KEY (User_ID, Playlist_ID, Track_ID)
);
-- how many stars does each playlit have
DROP TABLE IF EXISTS playlist_stats;
CREATE TABLE playlist_stats (
    User_ID INTEGER(4) NOT NULL,
    Playlist_ID INTEGER(4) NOT NULL,
    Rating INTEGER(1) DEFAULT 0,
    Count INTEGER(4) DEFAULT 0,
    PRIMARY KEY (User_ID, Playlist_ID, Rating)
);
-- example join of the data
-- SELECT t.Name FROM tracks t inner join  playlists p on t.Track_ID = p.Track_ID  WHERE p.Playlist_ID=27884