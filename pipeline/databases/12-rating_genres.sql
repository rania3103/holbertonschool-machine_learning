--  a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
SELECT tg.name AS name, SUM(tsr.rate) AS rating
FROM tv_show_genres AS tsg
INNER JOIN tv_show_ratings AS tsr
ON tsr.show_id = tsg.show_id
INNER JOIN tv_genres AS tg
ON tsg.genre_id = tg.id
GROUP BY name
ORDER BY rating DESC;
