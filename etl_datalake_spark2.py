import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
import glob
from pyspark.sql import types as T

os.environ["SPARK_HOME"] = "/usr/lib/spark"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"

#config = configparser.ConfigParser()
#config.read('dl.cfg')

#os.environ['AWS_ACCESS_KEY_ID']=config['AWS_CREDS']['AWS_ACCESS_KEY_ID']
#os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_CREDS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Reads the data from the song data json file (input_data) using a schema,
    loads it into a spark dataframe,
    create two subset dataframes, songs and artists, using only a few columns and
    writes the data into two parquet files in a given directory (output_data)
    """
    schema_song_data = T.StructType([
    T.StructField("artist_id", T.StringType(), True),
    T.StructField("artist_latitude", T.DoubleType(), True),
    T.StructField("artist_location", T.StringType(), True),
    T.StructField("artist_longitude", T.DoubleType(), True),
    T.StructField("artist_name", T.StringType(), True),
    T.StructField("duration", T.DoubleType(), True),
    T.StructField("num_songs", T.LongType(), True),
    T.StructField("song_id", T.StringType(), True),
    T.StructField("title", T.StringType(), True),
    T.StructField("year", T.LongType(), True)
    ])
    # Load the json file into a spark dataframe
    song_data = spark.read.json(input_data, schema_song_data)

    # extract columns to create songs table - song_id, title, artist_id, year, duration
    songs_table = song_data.select("song_id", "title", "artist_id", "year", "duration").distinct()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("artist_id", "year").mode("overwrite").parquet(output_data+"/songs_d.parquet")

    # extract columns to create artists table
    artists_table = song_data.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude").distinct()

    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(output_data+"/artists_d.parquet")


def process_log_data(spark, input_data, output_data):
    """
    Reads the data from the user log json file (input_data) using a schema,
    loads it into a spark dataframe,
    create two subset dataframes, users, time, using only a few columns and
    writes the data into three parquet files in a given directory (output_data).
    The script also creates 3 temp tables from artist and songs parquet file, and a subset of user log json file
    and joins these 3 tables to create songplays parquet file.
    """

    schema_user_log = T.StructType([
    T.StructField("artist", T.StringType(), True),
    T.StructField("auth", T.StringType(), True),
    T.StructField("firstName", T.StringType(), True),
    T.StructField("gender", T.StringType(), True),
    T.StructField("itemInSession", T.StringType(), True),
    T.StructField("lastName", T.StringType(), True),
    T.StructField("length", T.DoubleType(), True),
    T.StructField("level", T.StringType(), True),
    T.StructField("location", T.StringType(), True),
    T.StructField("method", T.StringType(), True),
    T.StructField("page", T.StringType(), True),
    T.StructField("registration", T.DoubleType(), True),
    T.StructField("sessionId", T.LongType(), True),
    T.StructField("song", T.StringType(), True),
    T.StructField("status", T.LongType(), True),
    T.StructField("ts", T.LongType(), True),
    T.StructField("userAgent", T.StringType(), True),
    T.StructField("userId", T.StringType(), True)
    ])


    user_log = spark.read.json(input_data, schema_user_log)

    # filter by actions for song plays
    user_log_next_song = user_log.where(user_log.page=='NextSong')

    # extract columns for users table
    users_table = user_log_next_song.select("userId", "firstName", "lastName", "gender", "level").distinct()

    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(output_data+"/users_d.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp( (x/1000) ), T.TimestampType())
    get_hour = udf(lambda x: x.hour, T.IntegerType())
    get_day = udf(lambda x: x.day, T.IntegerType())
    get_week = udf(lambda x: x.isocalendar()[1], T.IntegerType())
    get_month = udf(lambda x: x.month, T.IntegerType())
    get_year = udf(lambda x: x.isocalendar()[0], T.IntegerType())
    get_weekday = udf(lambda x: x.isocalendar()[2], T.IntegerType())

    user_log_next_song = user_log_next_song.withColumn("timestamp", get_timestamp(user_log_next_song.ts))
    user_log_next_song = user_log_next_song.withColumn("hour", get_hour(get_timestamp(user_log_next_song.ts)))
    user_log_next_song = user_log_next_song.withColumn("day", get_day(get_timestamp(user_log_next_song.ts)))
    user_log_next_song = user_log_next_song.withColumn("week", get_week(get_timestamp(user_log_next_song.ts)))
    user_log_next_song = user_log_next_song.withColumn("month", get_month(get_timestamp(user_log_next_song.ts)))
    user_log_next_song = user_log_next_song.withColumn("year", get_year(get_timestamp(user_log_next_song.ts)))
    user_log_next_song = user_log_next_song.withColumn("weekday", get_weekday(get_timestamp(user_log_next_song.ts)))

    # extract columns to create time table
    time_table = user_log_next_song.select("ts", "timestamp", "hour", "day", "week", "month", "year", "weekday")

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data+"/time_d.parquet")

    # read in song data to use for songplays table
    songs_table = spark.read.parquet(output_data+'/songs_d.parquet')
    artists_table = spark.read.parquet(output_data+'/artists_d.parquet')

    user_log_next_song.createOrReplaceTempView("user_log_next_song_v")
    artists_table.createOrReplaceTempView("artists_table_v")
    songs_table.createOrReplaceTempView("songs_table_v")

    # extract columns from joined song and log datasets to create songplays table
    songplay_table=spark.sql(
      '''
      SELECT distinct se.ts songplay_id, se.ts start_time, se.userId user_id, se.level,s.song_id song_id,a.artist_id artist_id,
      se.sessionId session_id, se.location, se.userAgent user_agent, se.year, se.month
      FROM user_log_next_song_v se JOIN artists_table_v a ON se.artist = a.artist_name
      JOIN songs_table_v s ON se.song = s.title
      ''')

    # write songplays table to parquet files partitioned by year and month
    songplay_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data+"/songplay_f.parquet")


def main():
    spark = create_spark_session()

    input_data_user_log='s3a://udacity-dend/log-data/2018/11/*'
    input_data_song_data='s3a://udacity-dend/song-data/*/*/*/*'

    #input_data_user_log='s3a://udacity-dend/log-data/2018/11/2018-11-01-events.json'
    #input_data_song_data='s3a://udacity-dend/song-data/A/A/A/TRAAABD128F429CF47.json'
    output_data='s3a://emr-udacity-test'

    process_song_data(spark, input_data_song_data, output_data)
    process_log_data(spark, input_data_user_log, output_data)


if __name__ == "__main__":
    main()
