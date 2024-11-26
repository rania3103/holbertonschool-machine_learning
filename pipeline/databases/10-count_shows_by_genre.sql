-- a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT t_g.name AS genre, COUNT(t_s_g.genre_id) AS number_of_shows
FROM tv_show_genres AS t_s_g
INNER JOIN tv_genres AS t_g
ON t_s_g.genre_id = t_g.id
GROUP BY genre
HAVING number_of_shows > 0
ORDER BY number_of_shows DESC ;
