/*CREATE INDEX idx_fts_doc_vec_proj ON unics_cordis.projects USING gin(document_vectors);*/

/*UPDATE 
    unics_cordis.projects
SET 
    document_vectors = (to_tsvector(acronym) || to_tsvector(title) || to_tsvector(ec_fund_scheme)|| to_tsvector(framework_program));*/
	
/*SELECT * FROM unics_cordis.projects WHERE document_vectors @@ to_tsquery('H2020');*/


/*stop words are removed with plainto_tsquery default configuration, plain is just text, to_tsquery allows for operators*/
/*SELECT * from unics_cordis.projects where document_vectors @@ plainto_tsquery('biologie and mathematics');*/ /*returns all rows with biology and mathematics*/
/*SELECT * from unics_cordis.projects where document_vectors @@ plainto_tsquery('data science');*/ /*returns all rows with both data and science*/
/*SELECT * from unics_cordis.projects where document_vectors @@ to_tsquery('Nanostructure&biology');*/ /*returns all rows that have biology or mathematics*/

/*FOR SPELLING MISTAKES*/

/*installs pg_trgm extension*/
/*CREATE EXTENSION pg_trgm;*/

/*create index over text columns*/
/*CREATE INDEX index_projects_all_text
             ON unics_cordis.projects using gin ((acronym|| ' ' || title|| ' ' ||ec_fund_scheme|| ' ' ||framework_program) gin_trgm_ops);*/
			 
/*searches across all indexed columns in the database*/
/*SELECT *, similarity('biology', 'biologie') from unics_cordis.projects*/

/*Finds top results with spelling mistake*/
/*SELECT *, similarity((title), 'biologie') AS sml
  FROM unics_cordis.projects
  ORDER BY sml DESC;*/

/*finds top results despite spelling mistake*/
/*SELECT *, similarity((title), 'bgi data') AS sml
  FROM unics_cordis.projects
  ORDER BY sml DESC; */
/*Select * From unics_cordis.projects;

Select string_agg(word::text, ' | ') FROM (
	SELECT word, similarity(word, 'bgi data') AS sml
  	FROM unics_cordis.words
	WHERE similarity(word, 'bgi data') >= .5
  	ORDER BY sml DESC 
  	) as nested;

SELECT * from unics_cordis.projects where document_vectors @@ to_tsquery(
	(Select string_agg(word, ' | ') FROM (
		SELECT word, similarity(word, 'bgi data') AS sml
  		FROM unics_cordis.words
		WHERE similarity(word, 'bgi data') >= .5
  		ORDER BY sml DESC 
  		) as nested
	));*/

/*Creates trigram index on document vector column from projects table, creates a new "dictionar" table with trigrams from ts_vector stemmed words */
/*CREATE TABLE unics_cordis.words AS SELECT word FROM
        ts_stat('SELECT document_vectors FROM unics_cordis.projects');  */
/*CREATE INDEX words_idx ON unics_cordis.words USING GIN (word gin_trgm_ops);*/