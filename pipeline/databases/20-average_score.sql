-- a SQL script that creates a stored procedure that computes and store the average score for a student.
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(IN us_id INT)
BEGIN
    DECLARE avg_score FLOAT;
    SELECT AVG(score) INTO avg_score 
    FROM corrections
    WHERE user_id = us_id;
    UPDATE users
    SET average_score = IFNULL(avg_score, 0)
    WHERE id = us_id;
END$$
DELIMITER ;
