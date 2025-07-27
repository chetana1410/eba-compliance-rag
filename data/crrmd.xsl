<?xml version="1.0" encoding="UTF-8"?>
<!-- HERE BE DRAGONS -->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns="http://www.w3.org/1999/xhtml"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:xslFormatting="urn:xslFormatting"
  xmlns:fx="http://publications.europa.eu/formex"
  exclude-result-prefixes="fx">
  <xsl:output method="text" encoding="UTF-8" indent="no" />
  <xsl:strip-space elements="FORMULA EXPR FRACTION DIVIDEND DIVISOR EXPONENT"/>
  <xsl:preserve-space elements="DLIST DLIST.ITEM TERM DEFINITION"/>

  <xsl:variable name="lowercase" select="'abcdefghijklmnopqrstuvwxyz'" />
  <xsl:variable name="uppercase" select="'ABCDEFGHIJKLMNOPQRSTUVWXYZ'" />

  <xsl:template match="/">
    <xsl:apply-templates select="//ARTICLE"/>
  </xsl:template>

  <xsl:template match="PREAMBLE.INIT|VISA|GR.CONSID.INIT|PREAMBLE.FINAL">
    <p><xsl:apply-templates select="node()"/></p>
  </xsl:template>

  <xsl:template name="footnotes">
    <section class="footnotes">
      <ol>
        <xsl:for-each select="/CONS.ACT/CONS.DOC/ENACTING.TERMS/descendant::NOTE[@TYPE='FOOTNOTE']">
          <li id="cite-note-{@NOTE.ID}">
            <a href="#cite-ref-{@NOTE.ID}">^</a><xsl:text> </xsl:text>
            <xsl:apply-templates match="." />
          </li>
        </xsl:for-each>
      </ol>
    </section>
  </xsl:template>

  <xsl:template name="toc">
    <xsl:param name="division" />
    <li>
      <a href="#ch{position()}">
      <xsl:value-of select="$division/TITLE/TI" /><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="$division/TITLE/STI" />
      </a><br />

      <ul>
        <xsl:for-each select="$division/ARTICLE">
          <li>
            <a href="#art{number(@IDENTIFIER)}"><xsl:value-of select="TI.ART" /></a><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="STI.ART" />
          </li>
        </xsl:for-each>

        <xsl:for-each select="$division/DIVISION">
          <li>
            <a href="#{translate(normalize-space(concat(../TITLE/TI, TITLE/TI)), ' ', '')}">
              <xsl:value-of select="TITLE/TI" /><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="TITLE/STI" />
            </a>
          </li>
          
          <li>
            <xsl:for-each select="ARTICLE">
              <a href="#art{number(@IDENTIFIER)}"><xsl:value-of select="TI.ART" /></a><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="STI.ART" /><br />
            </xsl:for-each>
          </li>
        </xsl:for-each>
      </ul>
    </li>
  </xsl:template>

  <xsl:template match="/CONS.ACT/CONS.DOC/ENACTING.TERMS/DIVISION">
    <a href="#ch{position()}">
      <h2 id="ch{position()}">
        <xsl:value-of select="normalize-space(TITLE/TI)"/><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="TITLE/STI" />
      </h2>
    </a>

    <xsl:apply-templates select="ARTICLE" />
    <xsl:apply-templates select="DIVISION" />
  </xsl:template>

  <xsl:template match="/CONS.ACT/CONS.DOC/ENACTING.TERMS/DIVISION/DIVISION">
    <a href="#{translate(normalize-space(concat(../TITLE/TI, TITLE/TI)), ' ', '')}">
      <h2 id="{translate(normalize-space(concat(../TITLE/TI, TITLE/TI)), ' ', '')}"><xsl:value-of select="normalize-space(TITLE/TI)"/><xsl:text disable-output-escaping="yes"> <![CDATA[&ndash;]]> </xsl:text><xsl:value-of select="normalize-space(TITLE/STI)"/></h2>
    </a>

    <xsl:apply-templates select="ARTICLE" />
    <xsl:apply-templates select="DIVISION" />
  </xsl:template>

  <xsl:template match="TITLE/STI">
    <span class="sti-div"><xsl:value-of select="normalize-space(.)"/></span>
  </xsl:template>

  <xsl:template match="ARTICLE">

---ARTICLE-SPLIT---

IDENTIFIER:<xsl:value-of select="@IDENTIFIER"/>

TITLE:<xsl:value-of select="TI.ART"/>

SUBTITLE:<xsl:value-of select="STI.ART"/>

---CONTENT---

<xsl:apply-templates select="PARAG" />
<xsl:apply-templates select="ALINEA" />

  </xsl:template>

  <xsl:template match="PARAG">

---PARAG-SPLIT---

IDENTIFIER:<xsl:value-of select="@IDENTIFIER"/>

NUMBER:<xsl:apply-templates select="NO.PARAG"/>

---CONTENT---

<xsl:apply-templates select="ALINEA[1]"/>
<xsl:for-each select="ALINEA[position()>1]">

<xsl:apply-templates select="."/>
</xsl:for-each>

  </xsl:template>

  <xsl:template match="PARAG/ALINEA">
    <xsl:apply-templates select="node()"/>
  </xsl:template>

  <xsl:template match="PARAG/ALINEA/LIST|ARTICLE/ALINEA/LIST|ARTICLE/ALINEA/LIST/ITEM/NP/P/LIST">
<xsl:apply-templates select="node()"/>
  </xsl:template>

  <xsl:template match="PARAG/ALINEA/LIST/ITEM|ARTICLE/ALINEA/LIST/ITEM|ARTICLE/ALINEA/LIST/ITEM/NP/P/LIST/ITEM">
- <xsl:apply-templates select="node()"/>

  </xsl:template>

  <xsl:template match="ARTICLE/ALINEA">
<xsl:apply-templates select="node()"/>

  </xsl:template>

  <xsl:template match="TITLE/TI/P">
<xsl:value-of select="normalize-space(.)"/>
  </xsl:template>

  <xsl:template match="HT[@TYPE='ITALIC']">
*<xsl:apply-templates select="node()"/>*</xsl:template>

  <xsl:template match="HT[@TYPE='UC']">
<xsl:apply-templates select="node()"/>
  </xsl:template>

  <xsl:template match="HT[@TYPE='BOLD']">
**<xsl:apply-templates select="node()"/>**</xsl:template>

  <xsl:template match="HT[@TYPE='SUB']">
<sub><xsl:apply-templates select="node()"/></sub></xsl:template>

  <xsl:template match="HT[@TYPE='SUP']">
<sup><xsl:apply-templates select="node()"/></sup></xsl:template>

  <xsl:template match="NOTE[@TYPE='FOOTNOTE']">
  </xsl:template>

  <xsl:template match="QUOT.START">"</xsl:template>
  <xsl:template match="QUOT.END">"</xsl:template>

  <xsl:decimal-format name="european" decimal-separator=',' grouping-separator='&#160;' />
  <xsl:template match="FT[@TYPE='NUMBER']">
    <xsl:value-of select="format-number(text(), '#&#160;###,##;(#&#160;###,##)', 'european')" />
  </xsl:template>

  <xsl:template match="REF.DOC.OJ">
[<xsl:value-of select="." />](https://eur-lex.europa.eu/legal-content/EN/AUTO/?uri=OJ:<xsl:value-of select="@COLL"/>:<xsl:value-of select="substring(@DATE.PUB, 1, 4)"/>:<xsl:value-of select="@NO.OJ"/>:TOC)</xsl:template>

  <!-- Formula rendering templates -->
  <xsl:template match="FORMULA">
$$<xsl:apply-templates/>$$
</xsl:template>

  <xsl:template match="FRACTION">
\frac{<xsl:apply-templates select="DIVIDEND"/>}{<xsl:apply-templates select="DIVISOR"/>}</xsl:template>

  <xsl:template match="DIVIDEND | DIVISOR">
<xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="OP.MATH">
<xsl:choose>
<xsl:when test="@TYPE='PLUS'"> + </xsl:when>
<xsl:when test="@TYPE='MINUS'"> - </xsl:when>
<xsl:when test="@TYPE='MULT'"> \times </xsl:when>
<xsl:when test="@TYPE='DIV'"> / </xsl:when>
<xsl:otherwise><xsl:value-of select="@TYPE"/></xsl:otherwise>
</xsl:choose>
  </xsl:template>

  <xsl:template match="OP.CMP">
<xsl:choose>
<xsl:when test="@TYPE='EQ'"> = </xsl:when>
<xsl:when test="@TYPE='LT'"> &lt; </xsl:when>
<xsl:when test="@TYPE='GT'"> &gt; </xsl:when>
<xsl:when test="@TYPE='LE'"> \leq </xsl:when>
<xsl:when test="@TYPE='GE'"> \geq </xsl:when>
<xsl:otherwise> <xsl:value-of select="@TYPE"/> </xsl:otherwise>
</xsl:choose>
  </xsl:template>

  <xsl:template match="EXPONENT">
^{<xsl:apply-templates/>}</xsl:template>

  <xsl:template match="BASE | EXP">
<xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="EXPR[@TYPE='BRACKET']">
\left(<xsl:apply-templates/>\right)</xsl:template>

  <xsl:template match="EXPR[@TYPE='BRACE']">
(<xsl:apply-templates/>)</xsl:template>

  <xsl:template match="EXPR[@TYPE='SQBRACKET']">
\left[<xsl:apply-templates/>\right]</xsl:template>

  <xsl:template match="EXPR">
<xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="P">
<xsl:apply-templates/>

  </xsl:template>

  <xsl:template match="P[DLIST]">
<xsl:apply-templates/>

  </xsl:template>

  <!-- Handle function names -->
  <xsl:template match="text()[normalize-space(.) = 'min']">\min</xsl:template>
  <xsl:template match="text()[normalize-space(.) = 'max']">\max</xsl:template>

  <!-- Handle definition lists -->
  <xsl:template match="DLIST" priority="1">

<xsl:apply-templates select="DLIST.ITEM"/>
  </xsl:template>

  <xsl:template match="*[local-name()='DLIST']" priority="0">

<xsl:apply-templates select="*[local-name()='DLIST.ITEM']"/>
  </xsl:template>

  <xsl:template match="DLIST.ITEM">
- **<xsl:apply-templates select="TERM"/>**<xsl:choose><xsl:when test="../@SEPARATOR"> <xsl:value-of select="../@SEPARATOR"/> </xsl:when><xsl:otherwise>: </xsl:otherwise></xsl:choose><xsl:apply-templates select="DEFINITION"/>

  </xsl:template>

  <xsl:template match="TERM">
<xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="DEFINITION">
<xsl:apply-templates/>
  </xsl:template>

  <!-- Handle included formula elements -->
  <xsl:template match="INCL.ELEMENT[@CONTENT='FORMULA']">

*[Formula: <xsl:value-of select="@FILEREF"/>]*

  </xsl:template>

  <!-- Handle mathematical elements -->
  <xsl:template match="IND">
_{<xsl:apply-templates/>}</xsl:template>

  <xsl:template match="SUM">
\sum<xsl:apply-templates select="UNDER"/>
  </xsl:template>

  <xsl:template match="UNDER">
_{<xsl:apply-templates/>}</xsl:template>

  <!-- Handle tables -->
  <xsl:template match="TBL">

| <xsl:for-each select="CORPUS/ROW[1]/CELL">Column <xsl:value-of select="position()"/> | </xsl:for-each>
|<xsl:for-each select="CORPUS/ROW[1]/CELL"> --- |</xsl:for-each>
<xsl:apply-templates select="CORPUS/ROW"/>
  </xsl:template>

  <xsl:template match="ROW">
| <xsl:for-each select="CELL"><xsl:apply-templates select="."/> | </xsl:for-each>
  </xsl:template>

  <xsl:template match="CELL">
<xsl:apply-templates/>
  </xsl:template>

</xsl:stylesheet>

